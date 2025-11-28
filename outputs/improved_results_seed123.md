# Improved Run results — seed 123

- Date (UTC): 2025-11-28T00:49:48.025984+00:00

- Improved GA-GAM (knee) Final Test RMSE: 0.6663

- Improved GA-GAM (best_by_rmse) Final Test RMSE: 0.6523

- Improved GA-GAM (best_by_penalty) Final Test RMSE: 0.7130

- Decision Tree Test RMSE: 0.6961

- Baseline PyGAM Test RMSE: 0.6697

- Penalty (baseline): 0.4562

- Penalty (knee): 0.1450

- Penalty (best_by_rmse): 0.3986

- Penalty (best_by_penalty): 0.0497


## Generation Log

| Gen | Best RMSE | Average RMSE |
|---:|---:|---:|

| 0 | 0.594268 | 0.717333 |

| 1 | 0.589725 | 0.652319 |

| 2 | 0.580499 | 0.640293 |

| 3 | 0.580499 | 0.634401 |

| 4 | 0.580499 | 0.631138 |

| 5 | 0.580499 | 0.611083 |

| 6 | 0.580499 | 0.611828 |

| 7 | 0.578691 | 0.611321 |

| 8 | 0.578691 | 0.613388 |

| 9 | 0.578691 | 0.616238 |

| 10 | 0.578691 | 0.609311 |

| 11 | 0.578691 | 0.610337 |

| 12 | 0.578691 | 0.608797 |

| 13 | 0.578691 | 0.608887 |

| 14 | 0.578691 | 0.606750 |

| 15 | 0.578691 | 0.609205 |

| 16 | 0.578653 | 0.608925 |

| 17 | 0.578653 | 0.608597 |

| 18 | 0.578653 | 0.610486 |

| 19 | 0.578653 | 0.611513 |

| 20 | 0.578653 | 0.610876 |


## Model Structure Summaries

### GA (knee)

- **MedInc**: spline

- **HouseAge**: linear

- **AveRooms**: linear

- **AveBedrms**: linear

- **Population**: none

- **AveOccup**: none

- **Latitude**: spline

- **Longitude**: spline


### GA (best_by_rmse)

- **MedInc**: linear

- **HouseAge**: linear

- **AveRooms**: spline

- **AveBedrms**: none

- **Population**: spline

- **AveOccup**: spline

- **Latitude**: spline

- **Longitude**: spline


### GA (best_by_penalty)

- **MedInc**: linear

- **HouseAge**: spline

- **AveRooms**: linear

- **AveBedrms**: linear

- **Population**: none

- **AveOccup**: linear

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
    "scale": false,
    "knots": 15,
    "lambda": 0.49936932246970356
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
    "type": "none",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 18,
    "lambda": 0.6065998823060254
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 20,
    "lambda": 2.6155823601485344
  }
]

```

### GA (best_by_rmse)
```json

[
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
    "knots": 20,
    "lambda": 0.132274698473292
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
    "knots": 14,
    "lambda": 0.048642770000132014
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 14,
    "lambda": 0.06677555553865241
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 15,
    "lambda": 0.027277184329942108
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 20,
    "lambda": 0.01341073826636006
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
    "type": "spline",
    "scale": true,
    "knots": 9,
    "lambda": 9.144512804903716
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
  }
]

```
