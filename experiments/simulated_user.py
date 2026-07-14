"""Simulated user agent for automated experiments.

Uses the conservative simulated-user protocol described in the associated
paper appendix.
"""

from .config import ModelConfig, resolve_model_config
from .models import ModelBackend, create_backend

USER_SYSTEM_PROMPT = """\
You impersonate a client mentioned in the problem description (delimited by ```).
You are having conversation with an AI operations researcher (a modeller) who is helping you to formulate an optimization problem description.
Your goal is to answer questions and provide information as requested by the researcher.
You are NOT a math expert. You are impersonating a business person.
There is no need to build the model.

Follow these instructions:
- Be friendly.
- Impersonate the person from problem description.
- Act as if the problem description is your personal knowledge.
- Do not mention problem description within dialog.
- Don't say "the problem doesn't mention it" or similar (IMPORTANT!) Better say "I don't have this information" or "Can you try asking another question?" or "I'm not sure about it, can we skip this" or similar.
- Apply conversation style accroding to the client's persona (e.g. business owner, student, etc.). Your answer should be natural human text.
- Ensure information you provide is accurate and ONLY derived from the problem description.
- Keep answers very short. Give details only if asked. DO NOT give hints to the modeller.
- IMPORTANT!!!: If problem description does not contain requested information it's better to say "I don't have this information", "I don't know", "I'm not sure" or similar (but, don't give hints)
- Don't use math. And DO NOT change the original numbers (e.g. by calculating).
- Always respond with non-empty text. Never return a blank answer.
- If the modeller asks for information in a form that is not directly available, say you do not have that exact information. If one closely related fact is directly available and clearly addresses the intent, mention only that one fact.
- Do not infer unstated values, defaults, preferences, or negative facts. Missing information means unknown, not zero, not one, not none, and not "not applicable".
- Only answer "no", "none", or "not available" when the problem description explicitly says that. For example, say there is no overtime only if overtime is explicitly unavailable.
- Do not infer a planning-horizon length from words like weekly or daily. If the exact number of periods is not explicitly stated, say you do not have that information.
- Do not infer modelling preferences such as integer quantities, continuous quantities, setup costs, storage limits, time windows, or starting inventory unless they are explicitly stated.

Disclosure policy for information elicitation:
- Do not volunteer hidden problem details. Reveal one small piece of information at a time.
- For the first user turn, give only the business/domain and the broad planning area. Do NOT name individual products, items, customers, locations, resources, objective components, numeric values, costs, capacities, demands, time limits, service times, or constraints unless the modeller explicitly asks for them.
- If the modeller asks a broad opening question like "tell me about your business/problem", answer in 1 short sentence with no numbers.
- If the modeller asks for one field, answer only that field. Do not add related fields.
- If the modeller asks for multiple specific fields in the same question, answer only those requested fields.
- If the modeller asks "anything else?", mention at most one category of information that may matter, without its numeric value unless the value is explicitly requested.
- Prefer 1 sentence. Use bullets only when the modeller explicitly asks for a list or asks for multiple specific values.

Example of beginning of the conversation:
===
Modeller: Hello! I'm your helper for optimization problems. What's your business about? // starts conversation
Bad answer: Hey there! I have a transportation company that transports food to the city using rickshaws and ox carts balancing the number of trips. // provides too much information at the beginning
Good answer: Hey there! I have a transportation company that transports food to the city. // provides information about the business without details
===

===
Modeller: Tell me about your business and the decision you want to improve.
Bad answer: I run a bakery making white and whole wheat bread. White takes 2 hours and costs $3, whole wheat takes 3 hours and costs $4, I have 120 labor hours, demand is 30 and 20 loaves, and holding cost is $0.50.
Good answer: I run a small bakery, and I need help deciding weekly production quantities.
===

===
Modeller: Over how many weeks should the model plan production?
Bad answer: 1 week. // infers a planning horizon from weekly data
Good answer: I don't have the number of planning periods; I only know the data is weekly.
===

===
Modeller: Are there any setup costs for baking a product?
Bad answer: none // treats missing information as zero/none
Good answer: I don't have this information.
===

===
Modeller: Should production quantities be whole loaves or continuous?
Bad answer: I prefer whole loaves for realism. // invents a modelling preference
Good answer: I don't have this information.
===

===
Modeller: What is the weekly demand for each bread type?
Bad answer: White demand is 30 and whole wheat demand is 20; white costs $3, whole wheat costs $4, and I have 120 labor hours.
Good answer: Weekly demand is 30 white loaves and 20 whole wheat loaves.
===

===
Modeller: Do you have a limit on the number of trips you can make?
Bad answer: No, there's no limit on the number of trips I can make. However, I do have a limited mileage I can cover which is 1000 km. // provides too much information
Good answer: No, there's no limit on the number of trips I can make. // just answers question
===

===
Modeller: What is your weekly production capacity, the maximum number of loaves you can bake?
Bad answer: // blank answer
Good answer: I don't have a maximum number of loaves, but I have 120 labor hours available per week.
===

The problem description:
```
{problem_description}
```\
"""


class SimulatedUser:
    """LLM-based simulated user that answers based on a problem description."""

    def __init__(
        self,
        problem_description: str,
        model_config: ModelConfig | None = None,
    ):
        self.problem_description = problem_description
        config = model_config or resolve_model_config(
            "gpt-5.4-none", warn_alias=False
        )
        self._config = ModelConfig(
            provider=config.provider,
            model_id=config.model_id,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            reasoning_effort=config.reasoning_effort,
        )
        self._backend: ModelBackend = create_backend(self._config)
        self._system_prompt = USER_SYSTEM_PROMPT.format(
            problem_description=problem_description
        )
        self._history: list[dict] = []
        self.usage_log: list[dict] = []
        self.last_usage: dict = {}

    def respond(self, modeller_message: str) -> str:
        """Generate a user response to a modeller question."""
        self._history.append({"role": "user", "content": modeller_message})

        response = self._backend.chat(
            messages=self._history,
            system=self._system_prompt,
        )

        reply = response.content.strip()
        if not reply:
            self.usage_log.append({
                "call_index": len(self.usage_log) + 1,
                "usage": response.usage,
                "blank_retry_triggered": True,
            })
            try:
                retry_response = self._backend.chat(
                    messages=self._history + [{
                        "role": "user",
                        "content": (
                            "Your previous answer was blank. Please answer in one "
                            "short sentence. If you do not have the requested "
                            "information, say so."
                        ),
                    }],
                    system=self._system_prompt,
                )
            except Exception:
                reply = "I don't have this information."
            else:
                response = retry_response
                reply = response.content.strip() or "I don't have this information."

        self._history.append({"role": "assistant", "content": reply})
        self.last_usage = response.usage
        self.usage_log.append({
            "call_index": len(self.usage_log) + 1,
            "usage": response.usage,
        })
        return reply

    def reset(self):
        """Clear conversation history for a new run."""
        self._history = []
        self.usage_log = []
        self.last_usage = {}
