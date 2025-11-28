# Improved Run results — seed 7

- Date (UTC): 2025-11-28T00:38:51.556274+00:00

- Improved GA-GAM (knee) Final Test RMSE: 0.6745

- Improved GA-GAM (best_by_rmse) Final Test RMSE: 0.6554

- Improved GA-GAM (best_by_penalty) Final Test RMSE: 0.7246

- Decision Tree Test RMSE: 0.7159

- Baseline PyGAM Test RMSE: 0.6674

- Penalty (baseline): 0.4384

- Penalty (knee): 0.1434

- Penalty (best_by_rmse): 0.2390

- Penalty (best_by_penalty): 0.0512


## Generation Log

| Gen | Best RMSE | Average RMSE |
|---:|---:|---:|

| 0 | 0.598216 | 0.725778 |

| 1 | 0.584065 | 0.666433 |

| 2 | 0.584065 | 0.667919 |

| 3 | 0.574862 | 0.642752 |

| 4 | 0.574862 | 0.639325 |

| 5 | 0.567599 | 0.619161 |

| 6 | 0.559003 | 0.612957 |

| 7 | 0.559003 | 0.616560 |

| 8 | 0.557052 | 0.613685 |

| 9 | 0.557052 | 0.618499 |

| 10 | 0.557052 | 0.623113 |

| 11 | 0.557052 | 0.623785 |

| 12 | 0.557052 | 0.630938 |

| 13 | 0.557052 | 0.646067 |

| 14 | 0.557052 | 0.608919 |

| 15 | 0.557052 | 0.603285 |

| 16 | 0.553778 | 0.602320 |

| 17 | 0.553778 | 0.601944 |

| 18 | 0.552070 | 0.602491 |

| 19 | 0.552070 | 0.599383 |

| 20 | 0.552070 | 0.594246 |


## Model Structure Summaries

### GA (knee)

- **MedInc**: spline

- **HouseAge**: none

- **AveRooms**: linear

- **AveBedrms**: linear

- **Population**: linear

- **AveOccup**: none

- **Latitude**: spline

- **Longitude**: spline


### GA (best_by_rmse)

- **MedInc**: spline

- **HouseAge**: linear

- **AveRooms**: linear

- **AveBedrms**: linear

- **Population**: linear

- **AveOccup**: spline

- **Latitude**: spline

- **Longitude**: spline


### GA (best_by_penalty)

- **MedInc**: linear

- **HouseAge**: spline

- **AveRooms**: linear

- **AveBedrms**: linear

- **Population**: linear

- **AveOccup**: none

- **Latitude**: linear

- **Longitude**: linear


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
    "type": "spline",
    "scale": true,
    "knots": 14,
    "lambda": 1.1805053645467793
  },
  {
    "type": "none",
    "scale": true,
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
    "type": "linear",
    "scale": true,
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
    "type": "spline",
    "scale": true,
    "knots": 19,
    "lambda": 0.8734338571943014
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 20,
    "lambda": 1.2889807783409777
  }
]

```

### GA (best_by_rmse)
```json

[
  {
    "type": "spline",
    "scale": false,
    "knots": 20,
    "lambda": 0.2031131381255136
  },
  {
    "type": "linear",
    "scale": true,
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
    "type": "linear",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 18,
    "lambda": 0.8103220160547339
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 20,
    "lambda": 0.0314700098723779
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 1.1776858120323452
  }
]

```

### GA (best_by_penalty)
```json

[
  {
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 12,
    "lambda": 6.0877652860106535
  },
  {
    "type": "linear",
    "scale": false,
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
    "type": "linear",
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
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": true,
    "knots": null,
    "lambda": null
  }
]

```
