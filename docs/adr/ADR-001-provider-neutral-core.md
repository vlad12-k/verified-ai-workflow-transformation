# ADR-001: Provider-Neutral Verification Core

**Status:** Accepted

## Context

VAIT must evaluate transformations across heterogeneous AI and non-AI
implementations without coupling its verification semantics to a single
model provider, cloud platform, or runtime.

A provider-specific core would make benchmark results harder to compare,
increase vendor lock-in, and mix execution concerns with transformation
verification.

IBM watsonx is valuable as a reference enterprise integration, but VAIT
must remain usable with other hosted providers and local models.

## Decision

The VAIT verification core is provider-neutral.

Core packages such as contracts, verification, decision, evaluation, and
optimisation must not directly depend on IBM, OpenAI, Anthropic, or other
provider SDKs.

Execution is exposed through stable runner interfaces.

Provider-specific implementations must be added as adapters behind those
interfaces.

Initial execution paths are expected to include:

- deterministic Python implementations;
- IBM watsonx;
- OpenAI-compatible providers;
- local Hugging Face models.

## Consequences

Benefits:

- verification semantics remain independent of model vendors;
- the same transformation contract can be evaluated across providers;
- provider integrations can evolve without changing the verification core;
- local and hosted implementations can participate in the same evaluation
  architecture;
- benchmark results are easier to reproduce and compare.

Costs:

- provider responses must be normalised into common execution observations;
- provider-specific metadata may require extension fields or adapter logic;
- abstraction boundaries must be maintained as integrations grow.

## Rejected Alternatives

### IBM-only core

Rejected because IBM is a reference integration rather than the semantic
boundary of the system.

### OpenAI-compatible API as the core abstraction

Rejected because not every implementation is an LLM or exposes an
OpenAI-compatible API.

### Provider SDK calls inside verification modules

Rejected because execution concerns would become coupled to verification
and decision logic.
