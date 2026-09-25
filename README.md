# GreetingIntelligenceOrchestrationPlatform (GIOP)

An enterprise-grade, cloud-native\*, AI-powered\*\* agent that determines whether to say "hello".

\* not actually cloud-native
\*\* contains zero (0) machine learning, but now has a neural network (it is a SHA-256 hash)

## Architecture

| Layer | Purpose | Why it's there |
|---|---|---|
| `IntentClassificationStrategy` | Decides if an utterance wants a greeting | Because an `if` statement needed an interface |
| `FarewellIntentClassificationStrategy` | Decides if an utterance wants a goodbye | The second `if` statement, with its own factory method and command |
| `NeuralGreetingClassifier` | Wraps a strategy and produces a confidence score | A 3-layer "deep neural network" that is a SHA-256 digest and some sigmoids |
| `AbstractGreetingComponentFactory` | Produces the strategy | An abstract factory, for the one strategy that exists |
| `GreetingComponentFactoryProvider` | Thread-safe singleton that produces the factory | The factory needed a provider |
| `InversionOfControlContainer` / `ContainerBootstrapper` / `ContainerBootstrapperFactory` | Resolves the provider that provides the factory | Direct instantiation was too direct |
| `EventBus` + `LoggingTelemetryObserver` | Publishes lifecycle events | Pub/sub for three log lines |
| `Command` / `CommandFactory` | Turns a decision into a response | GoF Command pattern for `f"Hello, {name}!"` |
| `CommandInterceptorChain` (`LoggingCommandInterceptor`, `TimingCommandInterceptor`) | Wraps command execution in middleware | Measures the microseconds spent returning `""` |
| `GreetingDecisionLedger` / `LedgerMiningObserver` | Mines every decision into a proof-of-work block chain | Blockchain. Difficulty 4, in memory, tamper-evident, single node |
| `UtteranceIngestionQueueWorker` | Background thread consuming a queue | Horizontally scales to one thread |

## Usage

```bash
python3 overengineered_agent.py
```

Feeds it `"hey there"`, `"quiet please"`, `"asdkjaslkdj"`, and `"goodbye now"`. Only the first and last produce output: `Hello, Bob!` and `Goodbye, Bob!`. Every decision is then mined into the ledger, which reports `valid=True`.

## Roadmap

- [ ] Kubernetes
- [x] A second `if` statement
- [x] Blockchain
