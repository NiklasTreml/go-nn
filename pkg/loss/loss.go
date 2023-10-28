package loss

import "gonum.org/v1/gonum/mat"

type LossFn func(yTrue, yPred *mat.Dense) float64
type LossPrimeFn func(yTrue, yPred *mat.Dense) *mat.Dense
