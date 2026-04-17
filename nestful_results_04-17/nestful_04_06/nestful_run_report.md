# Nestful Run Report

## Non-PTC

### Overview

| Metric | Value |
| --- | --- |
| Traces | 1861 |
| Scored traces | 1860 |
| Unscored traces | 1 |

### Token Usage

| Metric | Value |
| --- | --- |
| Total token usages | 3711030 |
| Input tokens | 1456726 |
| Output tokens | 63456 |
| Thinking tokens | 2190848 |
| Average total token usages | 1994.105 |
| Average input tokens | 782.765 |
| Average output tokens | 34.098 |
| Average thinking tokens | 1177.242 |
| Average planned tool calls | 1.073 |
| Average token usages for correct win rate | 2021.190 |
| Average token usages for correct full match | N/A |
| Average token usages for tasks both PTC and non-PTC solve | 1970.368 |
| Shared failed tasks (win rate = 0) | 397 |
| Average token usages for tasks both PTC and non-PTC fail on | 2224.446 |
| Average planned tool calls for tasks both PTC and non-PTC fail on | 1.146 |

### Latency

| Metric | Value |
| --- | --- |
| Average latency | 13.866 |
| Average latency for correct win rate | 10.709 |
| Average latency for full match | N/A |

### Quality Metrics

| Metric | Value |
| --- | --- |
| Runtime Errors | 2 |
| Parsing Error | 0 |
| Partial Match Accuracy | 0.165 |
| Full Match Accuracy | 0.000 |
| Win Rate | 0.034 |

### Error Analysis

| Metric | Value |
| --- | --- |
| wrong_first_tool | 746 |
| wrong_first_arguments | 728 |
| starts_correct_then_wrong_tool | 1 |
| starts_correct_then_wrong_arguments | 39 |
| stopped_early | 346 |
| extra_calls | 0 |
| parsing_error | 0 |
| pred_contains_unexpected_tool | 124 |
| pred_contains_unexpected_argument | 129 |

## PTC

### Overview

| Metric | Value |
| --- | --- |
| Traces | 1861 |
| Scored traces | 1861 |
| Unscored traces | 0 |

### Token Usage

| Metric | Value |
| --- | --- |
| Total token usages | 4996386 |
| Input tokens | 2755943 |
| Output tokens | 224379 |
| Thinking tokens | 2016064 |
| Average total token usages | 2684.786 |
| Average input tokens | 1480.894 |
| Average output tokens | 120.569 |
| Average thinking tokens | 1083.323 |
| Average planned tool calls | 3.420 |
| Average token usages for correct win rate | 2567.490 |
| Average token usages for correct full match | 2738.411 |
| Average token usages for tasks both PTC and non-PTC solve | 2900.491 |
| Shared failed tasks (win rate = 0) | 397 |
| Average token usages for tasks both PTC and non-PTC fail on | 3106.662 |
| Average planned tool calls for tasks both PTC and non-PTC fail on | 3.715 |

### Latency

| Metric | Value |
| --- | --- |
| Average latency | 17.671 |
| Average latency for correct win rate | 16.522 |
| Average latency for full match | 14.063 |

### Quality Metrics

| Metric | Value |
| --- | --- |
| Runtime Errors | 37 |
| Parsing Error | 0 |
| Partial Match Accuracy | 0.373 |
| Full Match Accuracy | 0.264 |
| Win Rate | 0.783 |

### Error Analysis

| Metric | Value |
| --- | --- |
| wrong_first_tool | 702 |
| wrong_first_arguments | 346 |
| starts_correct_then_wrong_tool | 132 |
| starts_correct_then_wrong_arguments | 122 |
| stopped_early | 61 |
| extra_calls | 7 |
| parsing_error | 0 |
| pred_contains_unexpected_tool | 295 |
| pred_contains_unexpected_argument | 305 |
