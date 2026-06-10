# ???????

- ????: 2026-06-11 00:13:42
- ????: `configs/dataset_build.yml`
- raw_images: 7722
- raw_labels: 7483
- paired_before_dedupe: 3987
- kept_samples: 3765
- unlabel_images: 216
- repeated_image_label_conflict_groups: 5
- num_classes: 58
- total_instances: 22322

## Batch ??

| batch | raw_images | raw_labels | paired_before_dedupe | kept_after_dedupe | unlabel_images | issues |
|---|---:|---:|---:|---:|---:|---:|
| data1 | 3450 | 3261 | 3258 | 3036 | 193 | 3 |
| data2 | 3520 | 3492 | 0 | 0 | 0 | 3520 |
| data3 | 217 | 210 | 210 | 210 | 7 | 0 |
| data4 | 39 | 39 | 39 | 39 | 0 | 0 |
| data5 | 320 | 320 | 320 | 320 | 0 | 0 |
| data6_std | 176 | 161 | 160 | 160 | 16 | 1 |

## ???????

| batch | unlabel_images |
|---|---:|
| data1 | 193 |
| data3 | 7 |
| data6_std | 16 |

## Issue ??

| issue | count |
|---|---:|
| image_unreadable | 3492 |
| repeated_image_same_label_kept_one | 124 |
| repeated_image_same_classes_boxes_auto_resolved | 56 |
| repeated_image_same_name_class_superset_auto_resolved | 32 |
| unlabel_image_unreadable | 28 |
| repeated_image_label_conflict | 5 |
| empty_label | 4 |

## Top ??

| class_name | instances | image_files |
|---|---:|---:|
| BoltHead | 11245 | 3004 |
| BoltNut | 5192 | 1860 |
| CotterPin | 564 | 336 |
| abnormal | 545 | 269 |
| Clamp | 516 | 290 |
| IronWire | 437 | 430 |
| BoltNut_z | 431 | 136 |
| BoltHead_1_s | 333 | 68 |
| PullTab | 328 | 328 |
| Spring | 254 | 237 |
| BoltHead_1 | 254 | 61 |
| BoltNut_1 | 217 | 79 |
| Nozzle | 213 | 209 |
| LockSpring | 179 | 178 |
| StoneSweeper | 177 | 177 |
| BrakePad | 144 | 143 |
| OilDamper | 140 | 140 |
| Sander | 101 | 101 |
| OilCap | 94 | 87 |
| CotterPin_1 | 82 | 53 |
| BrakeClamp | 70 | 70 |
| crack | 70 | 53 |
| AxleBox | 61 | 61 |
| SandBoxCover | 57 | 56 |
| CotterPin_k | 57 | 52 |
| oilLeak | 51 | 39 |
| IronWire_1 | 48 | 24 |
| AutomaticSplittingPhase | 35 | 35 |
| BrakeCylinder | 34 | 34 |
| WareHouseSocket | 31 | 31 |

## ????

- `docs\dataset_overall_batch_summary.csv`
- `docs\dataset_overall_class_stats.csv`
