# Improved Run results — seed 729

- Date (UTC): 2025-11-28T03:25:48.024847+00:00

- Improved GA-GAM (knee) Final Test RMSE: 0.6890

- Improved GA-GAM (best_by_rmse) Final Test RMSE: 0.6496

- Improved GA-GAM (best_by_penalty) Final Test RMSE: 0.7108

- Decision Tree Test RMSE: 0.6801

- Baseline PyGAM Test RMSE: 0.6520

- Penalty (baseline): 0.4608

- Penalty (knee): 0.0689

- Penalty (best_by_rmse): 0.2508

- Penalty (best_by_penalty): 0.0491


## Generation Log

| Gen | Best RMSE | Average RMSE |
|---:|---:|---:|

| 0 | 0.604890 | 0.788271 |

| 1 | 0.604890 | 0.691955 |

| 2 | 0.601952 | 0.670363 |

| 3 | 0.600180 | 0.652025 |

| 4 | 0.593513 | 0.651298 |

| 5 | 0.582714 | 0.628496 |

| 6 | 0.577280 | 0.620477 |

| 7 | 0.577280 | 0.621615 |

| 8 | 0.577280 | 0.612206 |

| 9 | 0.577280 | 0.611574 |

| 10 | 0.577280 | 0.613134 |

| 11 | 0.577280 | 0.612581 |

| 12 | 0.576467 | 0.614178 |

| 13 | 0.575754 | 0.617641 |

| 14 | 0.574339 | 0.617264 |

| 15 | 0.574339 | 0.615585 |

| 16 | 0.574339 | 0.614160 |

| 17 | 0.574339 | 0.613132 |

| 18 | 0.574339 | 0.613336 |

| 19 | 0.574339 | 0.612425 |

| 20 | 0.574339 | 0.610690 |


## Model Structure Summaries

### GA (knee)

- **MedInc**: linear

- **HouseAge**: linear

- **AveRooms**: none

- **AveBedrms**: linear

- **Population**: linear

- **AveOccup**: none

- **Latitude**: linear

- **Longitude**: spline


### GA (best_by_rmse)

- **MedInc**: spline

- **HouseAge**: spline

- **AveRooms**: linear

- **AveBedrms**: spline

- **Population**: linear

- **AveOccup**: none

- **Latitude**: spline

- **Longitude**: spline


### GA (best_by_penalty)

- **MedInc**: linear

- **HouseAge**: spline

- **AveRooms**: linear

- **AveBedrms**: linear

- **Population**: linear

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
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 19,
    "lambda": 2.201203737140892
  }
]

```

### GA (best_by_rmse)
```json

[
  {
    "type": "spline",
    "scale": false,
    "knots": 19,
    "lambda": 0.45045331175887277
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 18,
    "lambda": 5.614132250377772
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
    "knots": 11,
    "lambda": 6.378002384980165
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
    "type": "spline",
    "scale": false,
    "knots": 19,
    "lambda": 0.3867503861477974
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 19,
    "lambda": 0.28473591341221705
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
    "scale": false,
    "knots": 8,
    "lambda": 11.196737344751552
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
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  }
]

```
