# Project Changelog

A record of all major changes made to the project, referenced directly by commits.

Commit [68e77ee](https://github.com/lemoelink/LeMoE/commit/68e77ee)
- Integrated class-level threading locks for singletons to prevent race conditions during concurrent access.
- Implemented atomic configuration saving with temporary files to avoid config file corruption.
- Stabilized softmax calculations numerically to prevent exponential overflows at low temperatures.
- Added file name validation and namespace isolation for dynamic plugin loading.
- Integrated runtime type guards on plugin hooks to prevent None-propagation.

Commit [f95b9b0](https://github.com/lemoelink/LeMoE/commit/f95b9b0)
- Implemented cascading contextual routing architecture.
- Added silent self-correction features for expert routing.
- Integrated an adversarial test suite for robust validation.

Commit [21d935e](https://github.com/lemoelink/LeMoE/commit/21d935e)
- Added mandatory fallback model capability to ensure continuous service availability.
- Implemented an integrated update checker for the application.

Commit [4d1a7b9](https://github.com/lemoelink/LeMoE/commit/4d1a7b9)
- Completed the production-ready migration by transitioning the backend server to Gunicorn.
- Implemented core security hardening measures.
- Applied bug fixes and completed general codebase cleanup.

Commit [e237044](https://github.com/lemoelink/LeMoE/commit/e237044)
- Fixed Groq model deprecation issues.
- Optimized and lowered softmax temperature calculation parameters for 15 experts.

Commit [4962684](https://github.com/lemoelink/LeMoE/commit/4962684)
- Fixed router logic by lowering the softmax temperature to 0.02.
- Removed keyword threshold halving and cleaned up boundary underscores.

Commit [50ae5cb](https://github.com/lemoelink/LeMoE/commit/50ae5cb)
- Fixed a routing bug by stripping markdown formatting before evaluating user prompts.

Commit [9217be6](https://github.com/lemoelink/LeMoE/commit/9217be6)
- Created the initial public release of LEMoE - Light Easy Mix Of Experts.
