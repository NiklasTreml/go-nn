package layers

import "gonum.org/v1/gonum/mat"

type Layer interface {
	Forward(input *mat.Dense) *mat.Dense
	Backward(outputError *mat.Dense, alpha float64) *mat.Dense
}
