import torch
from copy import deepcopy
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.utils import is_apex_available, logging
from transformers.trainer_pt_utils import nested_detach
from transformers.generation import GenerationConfig
from transformers.integrations import is_deepspeed_zero3_enabled

    
from .base import BaseTrainer   

if TYPE_CHECKING:
    from transformers.data import DataCollator
    from transformers import (
        PreTrainedTokenizerBase,
        PreTrainedModel,
        TrainerCallback,
        EvalPrediction, 
        TrainingArguments
    )


logger = logging.get_logger(__name__)



class DualTrainer(BaseTrainer):    
    def training_step(self, model: nn.Module, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]]) -> torch.Tensor:
        """
        Perform training step 

        Args:
            model (Module): _description_
            inputs (Dict[str, Tensor  |  Any]): _description_

        Returns:
            Tensor: _description_
        """
        model.train()
        inputs = self._prepare_input(inputs) # inputs = (inputs[1], inputs[2])

        # Custom loss function 
        assert len(inputs) == len(self.losses)
        with self.compute_loss_context_manager():  
            if self.losses is not None:          
                opt_losses = [
                    self.compute_loss(model, inputs[idx], self.losses[idx]) for idx in range(len(self.losses))
                ]
                    
                loss = sum([l.weight * v for l, v in zip(self.losses, opt_losses)])
            else:
                loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def prediction_step(
        self, 
        model: nn.Module, 
        inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation step on 'model' using 'input'

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Tuple[Dict[str, Union[torch.Tensor, Any]]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Returns:
           Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
           logits and labels (each being optional).
        """
        inputs = self._prepare_input(inputs) # (inputs[0], inputs[1])
       
        if self.losses is None:  
            self.losses = [None] * len(inputs)
            
        assert len(inputs) == len(self.losses)
        
        outputs = [
            self._prediction_step(
                model, inputs[idx], 
                loss_fn = self.losses[idx],
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys
            ) for idx in range(len(self.losses))
        ]    
        loss = sum([l.weight * v[0] for l, v in zip(self.losses, outputs)])
       
        if prediction_loss_only:
            return (loss, None, None)

        logits = [item[1] for item in outputs]
        labels = [item[2] for item in outputs]

        return loss, logits, labels
        
    def _prediction_step(
        self,
        model: nn.Module, 
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        loss_fn = None,
        ignore_keys: Optional[List[str]] = None):
        
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, loss_fn, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    
    class Seq2SeqDualTrainer(BaseTrainer):
        def __init__(
            self,
            model: Union["PreTrainedModel", nn.Module] = None,
            args: "TrainingArguments" = None,
            data_collator: Optional["DataCollator"] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
            compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
            callbacks: Optional[List["TrainerCallback"]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        ):
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                model_init=model_init,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                optimizers=optimizers,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            )

            # Override self.model.generation_config if a GenerationConfig is specified in args.
            # Priority: args.generation_config > model.generation_config > default GenerationConfig.
            if self.args.generation_config is not None:
                gen_config = self.load_generation_config(self.args.generation_config)
                self.model.generation_config = gen_config

        @staticmethod
        def load_generation_config(gen_config_arg: Union[str, GenerationConfig]) -> GenerationConfig:
            """
            Loads a `~generation.GenerationConfig` from the `Seq2SeqTrainingArguments.generation_config` arguments.

            Args:
                gen_config_arg (`str` or [`~generation.GenerationConfig`]):
                    `Seq2SeqTrainingArguments.generation_config` argument.

            Returns:
                A `~generation.GenerationConfig`.
            """

            # GenerationConfig provided, nothing to do
            if isinstance(gen_config_arg, GenerationConfig):
                return deepcopy(gen_config_arg)

            # str or Path
            pretrained_model_name = Path(gen_config_arg) if isinstance(gen_config_arg, str) else gen_config_arg
            config_file_name = None

            # Figuring if it is path pointing to a file, pointing to a directory or else a model id or URL
            # This step is required in order to determine config_file_name
            if pretrained_model_name.is_file():
                config_file_name = pretrained_model_name.name
                pretrained_model_name = pretrained_model_name.parent
            # dir path
            elif pretrained_model_name.is_dir():
                pass
            # model id or URL
            else:
                pretrained_model_name = gen_config_arg

            gen_config = GenerationConfig.from_pretrained(pretrained_model_name, config_file_name)
            return gen_config
        
        def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs,
        ) -> Dict[str, float]:
            """
            Run evaluation and returns metrics.

            The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
            (pass it to the init `compute_metrics` argument).

            You can also subclass and override this method to inject custom behavior.

            Args:
                eval_dataset (`Dataset`, *optional*):
                    Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                    not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                    method.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.
                metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                    An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                    "eval_bleu" if the prefix is `"eval"` (default)
                max_length (`int`, *optional*):
                    The maximum target length to use when predicting with the generate method.
                num_beams (`int`, *optional*):
                    Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                    beam search.
                gen_kwargs:
                    Additional `generate` specific kwargs.

            Returns:
                A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
                dictionary also contains the epoch number which comes from the training state.
            """

            gen_kwargs = gen_kwargs.copy()

            # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
            # training args
            if (
                gen_kwargs.get("max_length") is None
                and gen_kwargs.get("max_new_tokens") is None
                and self.args.generation_max_length is not None
            ):
                gen_kwargs["max_length"] = self.args.generation_max_length
            if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
                gen_kwargs["num_beams"] = self.args.generation_num_beams
            self._gen_kwargs = gen_kwargs

            return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        def _prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            loss_fn = None,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
        ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                gen_kwargs:
                    Additional `generate` specific kwargs.

            Return:
                Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
                labels (each being optional).
            """

            if not self.args.predict_with_generate or prediction_loss_only:
                return super()._prediction_step(
                    model, inputs, loss_fn=loss_fn,
                    prediction_loss_only=prediction_loss_only, 
                    ignore_keys=ignore_keys
                )

            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)

            # Priority (handled in generate):
            # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
            if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
                gen_kwargs = self._gen_kwargs.copy()
            if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
                gen_kwargs.pop("num_beams")
            if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
                gen_kwargs.pop("max_length")

            default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
            gen_kwargs["synced_gpus"] = (
                gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
            )

            generation_inputs = inputs.copy()
            # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
            # (otherwise, it would continue generating from the padded `decoder_input_ids`)
            if (
                "labels" in generation_inputs
                and "decoder_input_ids" in generation_inputs
                and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
            ):
                generation_inputs = {
                    k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
                }
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

            # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
            # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
            # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
            if self.model.generation_config._from_model_config:
                self.model.generation_config._from_model_config = False

            # Retrieves GenerationConfig from model.generation_config
            gen_config = self.model.generation_config
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_config.max_length:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

            with torch.no_grad():
                if has_labels:
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if self.label_smoother is not None:
                        loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                else:
                    loss = None

            if self.args.prediction_loss_only:
                return loss, None, None

            if has_labels:
                labels = inputs["labels"]
                if labels.shape[-1] < gen_config.max_length:
                    labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
                elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                    labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
            else:
                labels = None

            return loss, generated_tokens, labels