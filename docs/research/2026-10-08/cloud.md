# Cloud execution and documentation research

Inspection date: 2026-10-08. Scope: M2, M7, M8, and project documents.
Use [the source manifest](cloud-sources.json) for dates, revisions, and evidence hashes.
This inspection did not allocate a runtime or start authentication.

## Current CLI versions

| Package | Latest published version | Publication date | Observed development revision |
| --- | --- | --- | --- |
| `google-colab-cli` | `0.7.4` | 2026-09-26 | `930275e62970df10055f411c8e28ca61755dcfc7`, 2026-10-06 |
| `kaggle` | `2.2.4` | 2026-07-23 | `9c57dd74c29904c40ff7879eed33644be89d2e2f`, 2026-10-07 |

The PyPI upload dates identify published packages.
Development commit dates do not identify new releases.
The existing project CLI versions match these published versions.
Sources: [CLOUD-01: Colab package](https://pypi.org/project/google-colab-cli/0.7.4/) and [CLOUD-04: Kaggle release](https://github.com/Kaggle/kaggle-cli/releases/tag/v2.2.4).

## Colab execution: M2 and M8

The Colab CLI supplies runtime allocation, execution, file retrieval, termination, session listing, and usage commands.
The released source supports TPU requests.
Source: [CLOUD-01: released CLI](https://github.com/googlecolab/google-colab-cli/blob/66d7963611bcce2d272488e982ad38e519c062f0/README.md).

The execution handler prints Jupyter error messages.
A normal command return therefore does not establish successful notebook computation.
Use a result receipt from the experiment, then retrieve its complete records.
Source: [CLOUD-02: execution handler](https://github.com/googlecolab/google-colab-cli/blob/66d7963611bcce2d272488e982ad38e519c062f0/src/colab_cli/commands/execution.py).

Colab hardware and runtime limits can change.
An accelerator request does not establish the hardware that executed the code.
Source: [CLOUD-03: Colab FAQ](https://research.google.com/colaboratory/faq.html).

Design decisions:

1. Keep the sealed runtime, source, inputs, and fixed tolerances for each comparison.
2. Record the actual TPU type and device count from the runtime.
3. Retrieve the result files before runtime termination.
4. Compare each file with its recorded byte count and hash.
5. Record termination, an empty owned session state, and zero active usage.

These requirements come from the project test plan.
The CLI documentation alone does not show that they passed.
The [October 4 record](../../changes/task2-colab-routes.md) supplies the latest project observations.

## Kaggle authentication and version identity: M8

The released CLI documents OAuth login, an API token, and legacy API credentials.
A browser session alone does not supply these CLI credentials.
Source: [CLOUD-05: authentication](https://github.com/Kaggle/kaggle-cli/blob/f0afa32699d28c97f82691728ada3ed8c16c5abf/docs/README.md#authentication).

The previous local inspection found no usable CLI credentials.
This research did not repeat that account inspection.
Authentication remains an execution dependency, not a source collection dependency.

The released kernel documentation describes `push` as an upload followed by execution.
It documents `output` and `status` for the latest run.
Listed accelerators can have account restrictions.
Source: [CLOUD-06: released kernel commands](https://github.com/Kaggle/kaggle-cli/blob/f0afa32699d28c97f82691728ada3ed8c16c5abf/docs/kernels.md).

The current development change log reports fixes for explicit kernel versions.
The released `kernels_status` parses the version but does not put it in its request.
Thus, a version suffix alone cannot establish the identity of the returned run in version `2.2.4`.
Sources: [CLOUD-07: development changes](https://github.com/Kaggle/kaggle-cli/blob/9c57dd74c29904c40ff7879eed33644be89d2e2f/CHANGELOG.md) and [released implementation](https://github.com/Kaggle/kaggle-cli/blob/f0afa32699d28c97f82691728ada3ed8c16c5abf/src/kaggle/api/kaggle_api_extended.py).

Design decisions:

1. Establish account access before the Kaggle execution step.
2. Record the returned kernel version and session identity.
3. Include an experiment identifier and source hashes inside every result receipt.
4. Compare returned records with the requested experiment before acceptance.
5. Compare the terminal version and Draft state with the required values.
6. Examine a fixed CLI correction before its use if explicit version retrieval is necessary.

Do not infer a released fix from a development change log.
Do not use `push` as a metadata inspection command.
The future version reader needs both matching and incorrect-version fixtures.
No Kaggle kernel was submitted during this research.

## Hardware comparison: M7 and M8

The current hardware pages list 16 GB HBM per v5e chip and 32 GB per v6e chip.
The pages show an update date of 2026-10-07.
These specifications do not establish notebook access or available memory during an experiment.
Sources: [CLOUD-08: v5e](https://docs.cloud.google.com/tpu/docs/v5e) and [v6e](https://docs.cloud.google.com/tpu/docs/v6e).

Compare the reference and Pallas paths on the same allocated TPU.
Record topology, precision, compiler, memory method, and all time samples.
Keep cross-provider numerical reproduction separate from performance comparisons across different hardware.

The Kaggle TPU page returned a JavaScript application shell during direct retrieval.
Its body was not available for source inspection.
The CLI source supplies the available command evidence; actual account access remains unverified.

## Writing standard

The official STE website still identifies Issue 9, dated 2025-01-15, as the current issue.
The current PDF matches the issue required by the project writing guide.
Source: [CLOUD-09: ASD-STE100](https://www.asd-ste100.org/).

Keep Issue 9 as the document standard.
Use the technical glossary for computer and mathematical terms.
Examine word meaning, grammar, and sentence length separately.
A sentence-length check does not establish full conformity or independent certification.

## Remaining tests

This research confirms source availability and selected source behavior.
It does not confirm current credentials, cloud allocation, version retrieval, or resource closure for a future run.
Complete those observations before acceptance of the corresponding experiment.
