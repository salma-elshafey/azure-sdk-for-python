# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os
import logging
from typing import Dict, Union, List, Optional

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._evaluators._common._validators import ToolDefinitionsValidator, ValidatorInterface
from ..._common.utils import reformat_conversation_history, reformat_agent_response, reformat_tool_definitions
from azure.ai.evaluation._model_configurations import Message
from azure.ai.evaluation._common._experimental import experimental

logger = logging.getLogger(__name__)


@experimental
class _TaskCompletionEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """The Task Completion evaluator determines whether an AI agent successfully completed the requested task based on:

        - Final outcome and deliverable of the task
        - Completeness of task requirements

    This evaluator focuses solely on task completion and success, not on task adherence or intent understanding.

    Scoring is binary:
    - 1 (pass): Task fully completed with usable deliverable that meets all user requirements
    - 0 (fail): Task incomplete, partially completed, or deliverable does not meet requirements

    The evaluation includes task requirement analysis, outcome assessment, and completion gap identification.


    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START task_completion_evaluator]
            :end-before: [END task_completion_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a _TaskCompletionEvaluator with a query and response.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START task_completion_evaluator]
            :end-before: [END task_completion_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a _TaskCompletionEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    """

    _PROMPTY_FILE = "task_completion.prompty"
    _RESULT_KEY = "task_completion"
    _OPTIONAL_PARAMS = ["tool_definitions"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/task_completion"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, **kwargs):
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)

        # Initialize input validator
        self._validator = ToolDefinitionsValidator(error_target=ErrorTarget.TASK_COMPLETION_EVALUATOR)

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            **kwargs,
        )

    @overload
    def __call__(
        self,
        *,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate task completion for a given query, response, and optionally tool definitions.
        The query and response can be either a string or a list of messages.


        Example with string inputs and no tools:
            evaluator = _TaskCompletionEvaluator(model_config)
            query = "Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine."
            response = "**Day 1:** Morning: Louvre Museum, Lunch: Le Comptoir du Relais..."

            result = evaluator(query=query, response=response)

        Example with list of messages:
            evaluator = _TaskCompletionEvaluator(model_config)
            query = [{'role': 'system', 'content': 'You are a helpful travel planning assistant.'}, {'createdAt': 1700000060, 'role': 'user', 'content': [{'type': 'text', 'text': 'Plan a 3-day Paris itinerary with cultural landmarks and cuisine'}]}]
            response = [{'createdAt': 1700000070, 'run_id': '0', 'role': 'assistant', 'content': [{'type': 'text', 'text': '**Day 1:** Morning: Visit Louvre Museum (9 AM - 12 PM)...'}]}]
            tool_definitions = [{'name': 'get_attractions', 'description': 'Get tourist attractions for a city.', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'The city name.'}}}}]

            result = evaluator(query=query, response=response, tool_definitions=tool_definitions)

        :keyword query: The query being evaluated, either a string or a list of messages.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response being evaluated, either a string or a list of messages (full agent response potentially including tool calls)
        :paramtype response: Union[str, List[dict]]
        :keyword tool_definitions: An optional list of messages containing the tool definitions the agent is aware of.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: A dictionary with the task completion evaluation results.
        :rtype: Dict[str, Union[str, float]]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Invokes the instance using the overloaded __call__ signature.

        For detailed parameter types and return value documentation, see the overloaded __call__ definition.
        """
        return super().__call__(*args, **kwargs)

    def _build_result(
        self,
        score: Optional[int],
        result: str,
        reason: str,
        status: str,
        details: Dict,
        prompty_output_dict: Optional[Dict] = None,
    ) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Build a standardized result dictionary.

        :param score: The evaluation score (1, 0, or None).
        :param result: The result label ("pass", "fail", "skipped", or "error").
        :param reason: The reasoning or explanation string.
        :param status: The evaluation status ("completed", "skipped", or "error").
        :param details: The properties/details dictionary.
        :param prompty_output_dict: Optional raw prompty output for extracting token metadata.
        :return: The standardized result dictionary.
        """
        p = prompty_output_dict if isinstance(prompty_output_dict, dict) else {}
        return {
            self._result_key: score,
            f"{self._result_key}_result": result,
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": status,
            f"{self._result_key}_details": details,
            f"{self._result_key}_prompt_tokens": p.get("input_token_count", 0),
            f"{self._result_key}_completion_tokens": p.get("output_token_count", 0),
            f"{self._result_key}_total_tokens": p.get("total_token_count", 0),
            f"{self._result_key}_finish_reason": p.get("finish_reason", ""),
            f"{self._result_key}_model": p.get("model_id", ""),
            f"{self._result_key}_sample_input": p.get("sample_input", ""),
            f"{self._result_key}_sample_output": p.get("sample_output", ""),
        }

    @override
    def _not_applicable_result(
        self, error_message: str, threshold: Union[int, float], has_details: bool = False
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the evaluation is not applicable (skipped)."""
        return self._build_result(
            score=None,
            result="skipped",
            reason=f"Not applicable: {error_message}",
            status="skipped",
            details={},
        )

    @override
    async def _real_call(self, **kwargs):
        """The asynchronous call where real end-to-end evaluation logic is performed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Validate input before processing
        self._validator.validate_eval_input(kwargs)

        return await super()._real_call(**kwargs)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do Task Completion evaluation.
        :param eval_input: The input to the evaluator. Expected to contain whatever inputs are needed for the _flow method
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        # Import helper functions from base class module
        from azure.ai.evaluation._evaluators._common._base_prompty_eval import (
            _is_intermediate_response,
            _preprocess_messages,
        )

        # we override the _do_eval method as we want the output to be a dictionary,
        # which is a different schema than _base_prompty_eval.py
        if "query" not in eval_input and "response" not in eval_input:
            raise EvaluationException(
                message=f"Both query and response must be provided as input to the Task Completion evaluator.",
                internal_message=f"Both query and response must be provided as input to the Task Completion evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.TASK_COMPLETION_EVALUATOR,
            )

        # Check for intermediate response
        if _is_intermediate_response(eval_input.get("response")):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
                has_details=True,
            )

        # Preprocess messages if they are lists
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        eval_input["query"] = reformat_conversation_history(eval_input["query"], logger, include_system_messages=True)
        eval_input["response"] = reformat_agent_response(eval_input["response"], logger, include_tool_messages=True)
        if "tool_definitions" in eval_input and eval_input["tool_definitions"]:
            eval_input["tool_definitions"] = reformat_tool_definitions(eval_input["tool_definitions"], logger)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if not isinstance(llm_output, dict):
            score = None
            result = "error"
            reasoning = "Evaluator returned invalid output."
            status = "error"
            properties = {}
        else:
            status = llm_output.get("status", "completed")
            reasoning = llm_output.get("reasoning", "")
            properties = llm_output.get("properties") or {}

            if status == "skipped":
                score = None
                result = "skipped"
            else:
                score_value = llm_output.get("score", 0)
                if isinstance(score_value, str):
                    score = 1 if score_value.strip() in ("1", "true") else 0
                elif isinstance(score_value, (int, float)):
                    score = 1 if score_value == 1 else 0
                else:
                    score = 1 if score_value else 0
                result = "pass" if score == 1 else "fail"

        return self._build_result(
            score=score,
            result=result,
            reason=reasoning,
            status=status,
            details=properties,
            prompty_output_dict=prompty_output_dict,
        )
