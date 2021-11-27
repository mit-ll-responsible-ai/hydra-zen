# A Road Map of hydra-zen's Development

This document details features and capabilities that we intend to add to hydra-zen in both the near-term and 
in the long-term.

## Road Map to `v0.4.0`

Hydra's promise of reproducibility -- that it will leave behind a YAML-serialized configuration of any job -- 
is a killer feature, especially for those who use Hydra to run experiments, simulations, etc. However, it is 
important that these configs can be validated/maintained, so that compatibility issues with old 
configs/results can be identified in an automated way. E.g., consider the case where a new release of a lab's 
own research code is incompatible with some aspects of an old experiment. Ideally, one could incorporate 
config-validation in their CI so that this issue is identified by nightly checks. Presently, one would likely 
need to manually re-run the experiment itself in order to catch this problem Thus hydra-zen `v0.4.0` is all 
about validating configs: be them YAML-based configs, manually defined dataclasses, or configs produced by 
hydra-zen's config-creation functions. 

### Strict validation performed by `hydra_zen.builds`, `hydra_zen.make_config`, et al.

(Added in [#163](https://github.com/mit-ll-responsible-ai/hydra-zen/pull/163))

hydra-zen's config-creation functions should provide strict validation that:
   1. All configured values are compatible with Hydra's supported primitive types.
   2. Fields specified in targeted configs are compatible with the target's signature

As of `v0.4.0rc1` hydra-zen provides runtime support for both of these. Additionally, hydra-zen's internal 
annotations also enables static tooling to flag unsupported config-value types.

Because these functions perform validation upon constructing the configs themselves, incorporating 
config-validation into ones CI / automated test suite is trivial: simply write a test that imports the 
resulting configs. This is generally inexpensive and does not actually require the configs to be instantiated.
Furthermore, any runtime validation errors will be localized to a particular config and thus will be easy to debug.

### Built-in support for additional primitive types

(Added in [#163](https://github.com/mit-ll-responsible-ai/hydra-zen/pull/163))

hydra-zen can elegantly provide support for a variety of common types that are not natively supported by 
Hydra. Thus values of types like `complex` and `pathlib.Path` can be specified directly within configs, and 
hydra-zen will automatically create targeted configs that "build" those values; the resulting structured 
configs carry no dependency on hydra-zen.

### Validation of general configs

hydra-zen should provide a `hydra_zen.validate` function that is capable of processing any yaml-string or 
structured config and validate that:
   1. All configured values are compatible with Hydra's supported primitive types.
   2. Fields specified in targeted configs are compatible with the target's signature.
   3. The annotations specified in structured configs are compatible with the subset of annotations that are natively supported by Hydra.
   4. **MAYBE** perform runtime type-checking of values against annotated types. In the case that a targeted config is specified as a value, we could "look ahead" at the type that will be produced via recursive instantiation. If this is something we want to enable, we would strongly prefer that we either leverage a third-party type-checker to handle this (e.g. pydantic or beartype), or that we utilize omegaconf/Hydra's internal type-checking utilities.

This will involve being able to recurse through nested configs.

One use case in mind here is that a user runs a Hydra job and wants to ensure that the job can be reproduced 
in the future against updated dependencies. Thus the user would thus add to their test suite a test that 
feeds the job's resulting YAML-based config to `hydra_zen.validate`. It should also be convenient for users to
validate their configs prior to actually launching their Hydra jobs.

`hydra_zen.validate` would likely expose various configurable options for the stages of validation -- and 
their "strictness levels" -- that will be performed. There is plenty of opportunity for additional tooling 
here, such as a pytest plugin, but we should stay focused on the essentials here. We would be open to outside 
contributions that provide this sort of additional polish.
