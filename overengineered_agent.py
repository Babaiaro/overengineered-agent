"""
GreetingIntelligenceOrchestrationPlatform (GIOP)
An enterprise-grade, cloud-native*, AI-powered** agent for saying "hello".

*  not actually cloud-native
** contains zero (0) machine learning
"""

from __future__ import annotations

import abc
import enum
import hashlib
import logging
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Domain layer: Value Objects
# ---------------------------------------------------------------------------

class IntentClassification(enum.Enum):
    GREETING_DESIRED = "GREETING_DESIRED"
    GREETING_UNDESIRED = "GREETING_UNDESIRED"
    FAREWELL_DESIRED = "FAREWELL_DESIRED"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class Utterance:
    text: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    received_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ClassificationPrediction:
    classification: IntentClassification
    confidence: float


@dataclass(frozen=True)
class AgentDecision:
    classification: IntentClassification
    confidence: float
    correlation_id: str


# ---------------------------------------------------------------------------
# Strategy layer: Intent Classification Strategies
# ---------------------------------------------------------------------------

@runtime_checkable
class IntentClassificationStrategy(Protocol):
    def classify(self, utterance: Utterance) -> IntentClassification: ...


class KeywordPresenceIntentClassificationStrategy(IntentClassificationStrategy):
    """Classifies intent via a sophisticated lexical substring analysis."""

    _POSITIVE_LEXICON = frozenset({"Hi", "Hello", "Hey", "Yo"})
    _NEGATIVE_LEXICON = frozenset({"No", "Quiet", "Silence"})

    def classify(self, utterance: Utterance) -> IntentClassification:
        tokens = set(utterance.text.lower().split())
        if tokens & self._NEGATIVE_LEXICON:
            return IntentClassification.GREETING_UNDESIRED
        if tokens & self._POSITIVE_LEXICON:
            return IntentClassification.GREETING_DESIRED
        return IntentClassification.INDETERMINATE


class FarewellIntentClassificationStrategy(IntentClassificationStrategy):
    """The second `if` statement. Detects departures via lexical substring analysis."""

    _FAREWELL_LEXICON = frozenset({"bye", "goodbye", "farewell", "cya"})

    def classify(self, utterance: Utterance) -> IntentClassification:
        tokens = set(utterance.text.lower().split())
        if tokens & self._FAREWELL_LEXICON:
            return IntentClassification.FAREWELL_DESIRED
        return IntentClassification.INDETERMINATE


@runtime_checkable
class ConfidenceAwareIntentClassificationStrategy(IntentClassificationStrategy, Protocol):
    def predict(self, utterance: Utterance) -> ClassificationPrediction: ...


class NeuralGreetingClassifier(ConfidenceAwareIntentClassificationStrategy):
    """
    A deep neural network (3 hidden layers, sigmoid activations) that wraps a strategy
    and produces a confidence score. The network is a SHA-256 digest. The weights are
    the laws of mathematics. It has never been trained, and it never needs to be.
    """

    _HIDDEN_LAYERS = 3

    def __init__(self, delegate: IntentClassificationStrategy) -> None:
        self._delegate = delegate

    def classify(self, utterance: Utterance) -> IntentClassification:
        return self.predict(utterance).classification

    def predict(self, utterance: Utterance) -> ClassificationPrediction:
        classification = self._delegate.classify(utterance)  # the actual decision, i.e. the `if`
        activation = self._forward_pass(utterance.text)
        if classification == IntentClassification.INDETERMINATE:
            confidence = 0.10 * activation
        else:
            confidence = 0.90 + 0.09 * activation
        return ClassificationPrediction(classification, round(confidence, 4))

    @classmethod
    def _forward_pass(cls, text: str) -> float:
        # Embedding layer: 32 bytes of SHA-256, normalised to [0, 1]
        activations = [b / 255 for b in hashlib.sha256(text.lower().encode()).digest()]
        # Hidden layers: each neuron fires on itself plus its neighbour, through a sigmoid
        for _ in range(cls._HIDDEN_LAYERS):
            width = len(activations)
            activations = [
                1 / (1 + math.exp(-(a + activations[(i + 1) % width] - 1)))
                for i, a in enumerate(activations)
            ]
        # Output layer: global average pooling
        return sum(activations) / len(activations)


class IntentClassificationStrategyFactory:
    """Factory for producing IntentClassificationStrategy instances."""

    @staticmethod
    def create_default_strategy() -> ConfidenceAwareIntentClassificationStrategy:
        return NeuralGreetingClassifier(KeywordPresenceIntentClassificationStrategy())

    @staticmethod
    def create_farewell_strategy() -> ConfidenceAwareIntentClassificationStrategy:
        return NeuralGreetingClassifier(FarewellIntentClassificationStrategy())


# ---------------------------------------------------------------------------
# Abstract Factory for the Abstract Factory (you read that right)
# ---------------------------------------------------------------------------

class AbstractGreetingComponentFactory(abc.ABC):
    @abc.abstractmethod
    def create_strategy(self) -> ConfidenceAwareIntentClassificationStrategy: ...

    @abc.abstractmethod
    def create_farewell_strategy(self) -> ConfidenceAwareIntentClassificationStrategy: ...


class DefaultGreetingComponentFactory(AbstractGreetingComponentFactory):
    def create_strategy(self) -> ConfidenceAwareIntentClassificationStrategy:
        return IntentClassificationStrategyFactory.create_default_strategy()

    def create_farewell_strategy(self) -> ConfidenceAwareIntentClassificationStrategy:
        return IntentClassificationStrategyFactory.create_farewell_strategy()


class GreetingComponentFactoryProvider:
    """Provides the factory that provides the strategy. Factories all the way down."""

    _instance: "GreetingComponentFactoryProvider | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def get_factory(self) -> AbstractGreetingComponentFactory:
        return DefaultGreetingComponentFactory()


# ---------------------------------------------------------------------------
# Inversion-of-Control layer: because instantiating a class directly
# would have been far too straightforward
# ---------------------------------------------------------------------------

class ServiceRegistrationKey(enum.Enum):
    GREETING_COMPONENT_FACTORY_PROVIDER = "GREETING_COMPONENT_FACTORY_PROVIDER"
    GREETING_DECISION_LEDGER = "GREETING_DECISION_LEDGER"


class ServiceRegistration:
    def __init__(self, factory: Callable[[], Any], singleton: bool = True) -> None:
        self._factory = factory
        self._singleton = singleton
        self._instance: Any | None = None

    def resolve(self) -> Any:
        if not self._singleton:
            return self._factory()
        if self._instance is None:
            self._instance = self._factory()
        return self._instance


class InversionOfControlContainer:
    """A registry of registrations that resolve providers that provide factories."""

    def __init__(self) -> None:
        self._registrations: dict[ServiceRegistrationKey, ServiceRegistration] = {}

    def register(self, key: ServiceRegistrationKey, factory: Callable[[], Any]) -> None:
        self._registrations[key] = ServiceRegistration(factory)

    def resolve(self, key: ServiceRegistrationKey) -> Any:
        return self._registrations[key].resolve()


class ContainerBootstrapper:
    """Bootstraps the container that resolves the provider that provides the factory."""

    @staticmethod
    def bootstrap() -> InversionOfControlContainer:
        container = InversionOfControlContainer()
        container.register(
            ServiceRegistrationKey.GREETING_COMPONENT_FACTORY_PROVIDER,
            GreetingComponentFactoryProvider,
        )
        container.register(
            ServiceRegistrationKey.GREETING_DECISION_LEDGER,
            GreetingDecisionLedger,
        )
        return container


class ContainerBootstrapperFactory:
    """Produces bootstrappers for containers. Someone had to."""

    @staticmethod
    def create() -> ContainerBootstrapper:
        return ContainerBootstrapper()


# ---------------------------------------------------------------------------
# Observer layer: Telemetry / Event Bus
# ---------------------------------------------------------------------------

class AgentEvent(enum.Enum):
    UTTERANCE_RECEIVED = "UTTERANCE_RECEIVED"
    DECISION_MADE = "DECISION_MADE"
    RESPONSE_EMITTED = "RESPONSE_EMITTED"


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[AgentEvent, list[Callable[[Any], None]]] = {}

    def subscribe(self, event: AgentEvent, handler: Callable[[Any], None]) -> None:
        self._subscribers.setdefault(event, []).append(handler)

    def publish(self, event: AgentEvent, payload: Any) -> None:
        for handler in self._subscribers.get(event, []):
            handler(payload)


class LoggingTelemetryObserver:
    """A dedicated microservice-shaped observer for structured observability."""

    def __init__(self, event_bus: EventBus) -> None:
        self._logger = logging.getLogger("GIOP.Telemetry")
        event_bus.subscribe(AgentEvent.UTTERANCE_RECEIVED, self._on_utterance)
        event_bus.subscribe(AgentEvent.DECISION_MADE, self._on_decision)
        event_bus.subscribe(AgentEvent.RESPONSE_EMITTED, self._on_response)

    def _on_utterance(self, payload: Utterance) -> None:
        self._logger.info("UTTERANCE_RECEIVED cid=%s text=%r", payload.correlation_id, payload.text)

    def _on_decision(self, payload: AgentDecision) -> None:
        self._logger.info(
            "DECISION_MADE cid=%s classification=%s confidence=%.2f",
            payload.correlation_id, payload.classification.value, payload.confidence,
        )

    def _on_response(self, payload: str) -> None:
        self._logger.info("RESPONSE_EMITTED text=%r", payload)


# ---------------------------------------------------------------------------
# Blockchain layer: an immutable, tamper-evident, proof-of-work-secured
# distributed(*) ledger of every greeting decision
#
# (*) in memory, on one machine, in one process
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LedgerBlock:
    index: int
    timestamp: float
    correlation_id: str
    classification: str
    confidence: float
    previous_hash: str
    nonce: int
    hash: str


class GreetingDecisionLedger:
    """Every decision is mined into a block, because 'hello' deserves consensus."""

    DIFFICULTY = 4  # required number of leading zero hex digits
    GENESIS_PREVIOUS_HASH = "0" * 64

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chain: list[LedgerBlock] = [self._mine(0, "GENESIS", "GENESIS", 0.0, self.GENESIS_PREVIOUS_HASH)]

    @staticmethod
    def _digest(index: int, timestamp: float, correlation_id: str, classification: str,
                confidence: float, previous_hash: str, nonce: int) -> str:
        payload = f"{index}|{timestamp!r}|{correlation_id}|{classification}|{confidence!r}|{previous_hash}|{nonce}"
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def _mine(cls, index: int, correlation_id: str, classification: str,
              confidence: float, previous_hash: str) -> LedgerBlock:
        timestamp = time.time()
        target = "0" * cls.DIFFICULTY
        nonce = 0
        while True:
            digest = cls._digest(index, timestamp, correlation_id, classification, confidence, previous_hash, nonce)
            if digest.startswith(target):
                return LedgerBlock(index, timestamp, correlation_id, classification,
                                   confidence, previous_hash, nonce, digest)
            nonce += 1

    def record(self, decision: AgentDecision) -> LedgerBlock:
        with self._lock:
            tip = self._chain[-1]
            block = self._mine(tip.index + 1, decision.correlation_id, decision.classification.value,
                               decision.confidence, tip.hash)
            self._chain.append(block)
            return block

    def is_valid(self) -> bool:
        with self._lock:
            target = "0" * self.DIFFICULTY
            for previous, block in zip([None, *self._chain], self._chain):
                expected = self._digest(block.index, block.timestamp, block.correlation_id, block.classification,
                                        block.confidence, block.previous_hash, block.nonce)
                if block.hash != expected or not block.hash.startswith(target):
                    return False
                expected_previous = previous.hash if previous else self.GENESIS_PREVIOUS_HASH
                if block.previous_hash != expected_previous:
                    return False
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._chain)


class LedgerMiningObserver:
    """Listens for decisions and mines them into the chain, so that no greeting goes unattested."""

    def __init__(self, event_bus: EventBus, ledger: GreetingDecisionLedger) -> None:
        self._logger = logging.getLogger("GIOP.Ledger")
        self._ledger = ledger
        event_bus.subscribe(AgentEvent.DECISION_MADE, self._on_decision)

    def _on_decision(self, payload: AgentDecision) -> None:
        block = self._ledger.record(payload)
        self._logger.info(
            "BLOCK_MINED index=%d nonce=%d hash=%s...", block.index, block.nonce, block.hash[:16]
        )


# ---------------------------------------------------------------------------
# Command layer
# ---------------------------------------------------------------------------

class Command(abc.ABC):
    @abc.abstractmethod
    def execute(self) -> Any: ...


class EmitGreetingCommand(Command):
    def __init__(self, recipient: str) -> None:
        self._recipient = recipient

    def execute(self) -> str:
        return f"Hello, {self._recipient}!"


class EmitFarewellCommand(Command):
    def __init__(self, recipient: str) -> None:
        self._recipient = recipient

    def execute(self) -> str:
        return f"Goodbye, {self._recipient}!"


class NoOpCommand(Command):
    def execute(self) -> str:
        return ""


class CommandFactory:
    @staticmethod
    def for_classification(classification: IntentClassification, recipient: str) -> Command:
        if classification == IntentClassification.GREETING_DESIRED:
            return EmitGreetingCommand(recipient)
        if classification == IntentClassification.FAREWELL_DESIRED:
            return EmitFarewellCommand(recipient)
        return NoOpCommand()


# ---------------------------------------------------------------------------
# Middleware layer: Command Interceptor Chain
# (a function call, mediated by a chain of objects that agree to call
#  the next object in the chain)
# ---------------------------------------------------------------------------

class CommandInterceptor(abc.ABC):
    @abc.abstractmethod
    def intercept(self, command: Command, proceed: Callable[[], Any]) -> Any: ...


class LoggingCommandInterceptor(CommandInterceptor):
    def __init__(self) -> None:
        self._logger = logging.getLogger("GIOP.Interceptor.Logging")

    def intercept(self, command: Command, proceed: Callable[[], Any]) -> Any:
        self._logger.info("Invoking command=%s", type(command).__name__)
        result = proceed()
        self._logger.info("Command=%s returned %r", type(command).__name__, result)
        return result


class TimingCommandInterceptor(CommandInterceptor):
    def __init__(self) -> None:
        self._logger = logging.getLogger("GIOP.Interceptor.Timing")

    def intercept(self, command: Command, proceed: Callable[[], Any]) -> Any:
        started_at = time.perf_counter()
        result = proceed()
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        self._logger.info("Command=%s took %.4fms", type(command).__name__, elapsed_ms)
        return result


class CommandInterceptorChain:
    """Wraps command.execute() in an arbitrarily deep onion of middleware."""

    def __init__(self, interceptors: list[CommandInterceptor]) -> None:
        self._interceptors = interceptors

    def invoke(self, command: Command) -> Any:
        def build(index: int) -> Callable[[], Any]:
            if index >= len(self._interceptors):
                return command.execute
            return lambda: self._interceptors[index].intercept(command, build(index + 1))

        return build(0)()


class CommandInterceptorChainFactory:
    @staticmethod
    def create_default_chain() -> CommandInterceptorChain:
        return CommandInterceptorChain([LoggingCommandInterceptor(), TimingCommandInterceptor()])


# ---------------------------------------------------------------------------
# Async work-queue layer (because a function call wasn't distributed enough)
# ---------------------------------------------------------------------------

class UtteranceIngestionQueueWorker(threading.Thread):
    def __init__(self, work_queue: "queue.Queue[Utterance]", on_item: Callable[[Utterance], None]) -> None:
        super().__init__(daemon=True)
        self._queue = work_queue
        self._on_item = on_item
        self._running = True

    def run(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._on_item(item)
            self._queue.task_done()

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# The Orchestration Platform itself
# ---------------------------------------------------------------------------

class GreetingIntelligenceOrchestrationPlatform:
    """
    A horizontally-scalable(*), fault-tolerant(**), AI-driven(***) agent
    that determines whether to say "hello".

    (*)   runs on one thread
    (**)  has no error handling
    (***) if-statements
    """

    def __init__(self, recipient: str = "World") -> None:
        self._recipient = recipient
        self._container = ContainerBootstrapperFactory.create().bootstrap()
        provider: GreetingComponentFactoryProvider = self._container.resolve(
            ServiceRegistrationKey.GREETING_COMPONENT_FACTORY_PROVIDER
        )
        self._factory = provider.get_factory()
        self._strategy = self._factory.create_strategy()
        self._farewell_strategy = self._factory.create_farewell_strategy()
        self._event_bus = EventBus()
        self._telemetry = LoggingTelemetryObserver(self._event_bus)
        self._ledger: GreetingDecisionLedger = self._container.resolve(
            ServiceRegistrationKey.GREETING_DECISION_LEDGER
        )
        self._ledger_observer = LedgerMiningObserver(self._event_bus, self._ledger)
        self._interceptor_chain = CommandInterceptorChainFactory.create_default_chain()
        self._ingestion_queue: "queue.Queue[Utterance]" = queue.Queue()
        self._results: list[str] = []
        self._worker = UtteranceIngestionQueueWorker(self._ingestion_queue, self._process)
        self._worker.start()

    @property
    def ledger(self) -> GreetingDecisionLedger:
        return self._ledger

    def submit(self, text: str) -> None:
        utterance = Utterance(text=text)
        self._event_bus.publish(AgentEvent.UTTERANCE_RECEIVED, utterance)
        self._ingestion_queue.put(utterance)

    def _process(self, utterance: Utterance) -> None:
        prediction = self._strategy.predict(utterance)
        if prediction.classification == IntentClassification.INDETERMINATE:
            prediction = self._farewell_strategy.predict(utterance)  # the second `if`
        decision = AgentDecision(
            classification=prediction.classification,
            confidence=prediction.confidence,
            correlation_id=utterance.correlation_id,
        )
        self._event_bus.publish(AgentEvent.DECISION_MADE, decision)

        command = CommandFactory.for_classification(decision.classification, self._recipient)
        response = self._interceptor_chain.invoke(command)
        if response:
            self._event_bus.publish(AgentEvent.RESPONSE_EMITTED, response)
            self._results.append(response)

    def shutdown(self) -> list[str]:
        self._ingestion_queue.join()
        self._worker.stop()
        return self._results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    platform = GreetingIntelligenceOrchestrationPlatform(recipient="Bob")
    platform.submit("hey there")
    platform.submit("quiet please")
    platform.submit("asdkjaslkdj")
    platform.submit("goodbye now")
    results = platform.shutdown()
    print("\nFinal agent output:", results)
    print(f"Ledger: {len(platform.ledger)} blocks, valid={platform.ledger.is_valid()}")


if __name__ == "__main__":
    main()
