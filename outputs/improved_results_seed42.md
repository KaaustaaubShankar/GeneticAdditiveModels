# Improved Run results — seed 42

- Date (UTC): 2025-11-28T00:28:51.807995+00:00

- Improved GA-GAM (knee) Final Test RMSE: 0.7125

- Improved GA-GAM (best_by_rmse) Final Test RMSE: 0.6821

- Improved GA-GAM (best_by_penalty) Final Test RMSE: 0.7400

- Decision Tree Test RMSE: 0.6925

- Baseline PyGAM Test RMSE: 0.7216

- Penalty (baseline): 0.4096

- Penalty (knee): 0.1020

- Penalty (best_by_rmse): 0.2226

- Penalty (best_by_penalty): 0.0495


## Generation Log

| Gen | Best RMSE | Average RMSE |
|---:|---:|---:|

| 0 | 0.587837 | 0.721758 |

| 1 | 0.587837 | 0.676549 |

| 2 | 0.586357 | 0.648155 |

| 3 | 0.584597 | 0.626682 |

| 4 | 0.578648 | 0.613690 |

| 5 | 0.578648 | 0.612452 |

| 6 | 0.577510 | 0.613521 |

| 7 | 0.574708 | 0.616434 |

| 8 | 0.574707 | 0.621607 |

| 9 | 0.572594 | 0.604937 |

| 10 | 0.572594 | 0.602173 |

| 11 | 0.571410 | 0.602485 |

| 12 | 0.571410 | 0.602451 |

| 13 | 0.571410 | 0.600299 |

| 14 | 0.571410 | 0.598784 |

| 15 | 0.571410 | 0.598616 |

| 16 | 0.571410 | 0.596786 |

| 17 | 0.571399 | 0.596938 |

| 18 | 0.571399 | 0.598245 |

| 19 | 0.571399 | 0.596729 |

| 20 | 0.571399 | 0.594836 |


## Model Structure Summaries

### GA (knee)

- **MedInc**: spline

- **HouseAge**: linear

- **AveRooms**: linear

- **AveBedrms**: linear

- **Population**: none

- **AveOccup**: none

- **Latitude**: linear

- **Longitude**: spline


### GA (best_by_rmse)

- **MedInc**: spline

- **HouseAge**: linear

- **AveRooms**: none

- **AveBedrms**: spline

- **Population**: none

- **AveOccup**: linear

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
    "type": "spline",
    "scale": false,
    "knots": 13,
    "lambda": 4.393922957225035
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
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 3.4751617263202097
  }
]

```

### GA (best_by_rmse)
```json

[
  {
    "type": "spline",
    "scale": true,
    "knots": 9,
    "lambda": 0.032624289482092714
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
    "knots": 10,
    "lambda": 2.686688657029766
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
    "scale": true,
    "knots": 20,
    "lambda": 1.8897217095187007
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 0.03093454626210925
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
    "lambda": 8.537005045492963
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
