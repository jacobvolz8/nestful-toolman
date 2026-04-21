# Nestful Run Report

## Non-PTC

### Overview

| Metric | Value |
| --- | --- |
| Traces | 1861 |
| Scored traces | 1861 |
| Unscored traces | 0 |

### Token Usage

| Metric | Value |
| --- | --- |
| Total token usages | 3700733 |
| Input tokens | 1456723 |
| Output tokens | 63018 |
| Thinking tokens | 2180992 |
| Average total token usages | 1988.572 |
| Average input tokens | 782.764 |
| Average output tokens | 33.862 |
| Average thinking tokens | 1171.946 |
| Average planned tool calls | 1.074 |
| Average token usages for correct win rate | 1982.453 |
| Average token usages for correct full match | N/A |
| Average token usages for tasks both PTC and non-PTC solve | 1925.581 |
| Shared failed tasks (win rate = 0) | 407 |
| Average token usages for tasks both PTC and non-PTC fail on | 2215.912 |
| Average planned tool calls for tasks both PTC and non-PTC fail on | 1.120 |

### Latency

| Metric | Value |
| --- | --- |
| Average latency | 17.506 |
| Average latency for correct win rate | 14.447 |
| Average latency for full match | N/A |

### Quality Metrics

| Metric | Value |
| --- | --- |
| Runtime Errors | 2 |
| Parsing Error | 0 |
| Partial Match Accuracy | 0.164 |
| Full Match Accuracy | 0.000 |
| Win Rate | 0.028 |

### Error Analysis

| Metric | Value |
| --- | --- |
| wrong_first_tool | 741 |
| wrong_first_arguments | 734 |
| starts_correct_then_wrong_tool | 4 |
| starts_correct_then_wrong_arguments | 27 |
| stopped_early | 355 |
| extra_calls | 0 |
| parsing_error | 0 |
| pred_contains_unexpected_tool | 136 |
| pred_contains_unexpected_argument | 141 |

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
| Total token usages | 5040231 |
| Input tokens | 2759360 |
| Output tokens | 232679 |
| Thinking tokens | 2048192 |
| Average total token usages | 2708.346 |
| Average input tokens | 1482.730 |
| Average output tokens | 125.029 |
| Average thinking tokens | 1100.587 |
| Average planned tool calls | 3.509 |
| Average token usages for correct win rate | 2577.372 |
| Average token usages for correct full match | 2737.136 |
| Average token usages for tasks both PTC and non-PTC solve | 2904.512 |
| Shared failed tasks (win rate = 0) | 407 |
| Average token usages for tasks both PTC and non-PTC fail on | 3163.135 |
| Average planned tool calls for tasks both PTC and non-PTC fail on | 3.931 |

### Latency

| Metric | Value |
| --- | --- |
| Average latency | 16.570 |
| Average latency for correct win rate | 15.411 |
| Average latency for full match | 13.479 |

### Quality Metrics

| Metric | Value |
| --- | --- |
| Runtime Errors | 36 |
| Parsing Error | 0 |
| Partial Match Accuracy | 0.377 |
| Full Match Accuracy | 0.269 |
| Win Rate | 0.776 |

### Error Analysis

| Metric | Value |
| --- | --- |
| wrong_first_tool | 692 |
| wrong_first_arguments | 344 |
| starts_correct_then_wrong_tool | 124 |
| starts_correct_then_wrong_arguments | 114 |
| stopped_early | 73 |
| extra_calls | 13 |
| parsing_error | 0 |
| pred_contains_unexpected_tool | 286 |
| pred_contains_unexpected_argument | 296 |
