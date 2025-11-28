# Improved Run results — seed 225

- Date (UTC): 2025-11-28T01:02:53.087556+00:00

- Improved GA-GAM (knee) Final Test RMSE: 0.6702

- Improved GA-GAM (best_by_rmse) Final Test RMSE: 0.6228

- Improved GA-GAM (best_by_penalty) Final Test RMSE: 0.7824

- Decision Tree Test RMSE: 0.6770

- Baseline PyGAM Test RMSE: 0.6536

- Penalty (baseline): 0.4353

- Penalty (knee): 0.1083

- Penalty (best_by_rmse): 0.2905

- Penalty (best_by_penalty): 0.0549


## Generation Log

| Gen | Best RMSE | Average RMSE |
|---:|---:|---:|

| 0 | 0.600391 | 0.723505 |

| 1 | 0.592106 | 0.669450 |

| 2 | 0.590108 | 0.646509 |

| 3 | 0.585338 | 0.631929 |

| 4 | 0.581319 | 0.626830 |

| 5 | 0.581319 | 0.629908 |

| 6 | 0.577195 | 0.617459 |

| 7 | 0.577195 | 0.612970 |

| 8 | 0.577195 | 0.612970 |

| 9 | 0.577195 | 0.613021 |

| 10 | 0.563646 | 0.612243 |

| 11 | 0.563646 | 0.610873 |

| 12 | 0.563646 | 0.610026 |

| 13 | 0.563646 | 0.612601 |

| 14 | 0.563646 | 0.613079 |

| 15 | 0.562959 | 0.614548 |

| 16 | 0.562959 | 0.613334 |

| 17 | 0.562959 | 0.614685 |

| 18 | 0.562959 | 0.614779 |

| 19 | 0.562959 | 0.614360 |

| 20 | 0.562959 | 0.612048 |


## Model Structure Summaries

### GA (knee)

- **MedInc**: linear

- **HouseAge**: linear

- **AveRooms**: none

- **AveBedrms**: linear

- **Population**: linear

- **AveOccup**: none

- **Latitude**: spline

- **Longitude**: spline


### GA (best_by_rmse)

- **MedInc**: spline

- **HouseAge**: linear

- **AveRooms**: none

- **AveBedrms**: linear

- **Population**: linear

- **AveOccup**: spline

- **Latitude**: spline

- **Longitude**: spline


### GA (best_by_penalty)

- **MedInc**: linear

- **HouseAge**: linear

- **AveRooms**: none

- **AveBedrms**: none

- **Population**: linear

- **AveOccup**: linear

- **Latitude**: spline

- **Longitude**: none


### Baseline PyGAM

- **MedInc_baseline**: spline

- **HouseAge_baseline**: spline

- **AveRooms_baseline**: spline

- **AveBedrms_baseline**: spline

- **Population_baseline**: spline

- **AveOccup_baseline**: spline

- **Latitude_baseline**: spline

- **Longitude_baseline**: spline


## Best Chromosomes (JSON)

### GA (knee)
```json

[
  {
    "type": "linear",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "none",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "none",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 19,
    "lambda": 1.7644728426106262
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 0.5894213075112351
  }
]

```

### GA (best_by_rmse)
```json

[
  {
    "type": "spline",
    "scale": true,
    "knots": 10,
    "lambda": 0.2746161590867651
  },
  {
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "none",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 17,
    "lambda": 0.03144432753003858
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 14,
    "lambda": 0.14830707942845933
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 0.5112256447846397
  }
]

```

### GA (best_by_penalty)
```json

[
  {
    "type": "linear",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "none",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "none",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 6,
    "lambda": 7.5129935299918635
  },
  {
    "type": "none",
    "scale": false,
    "knots": null,
    "lambda": null
  }
]

```
