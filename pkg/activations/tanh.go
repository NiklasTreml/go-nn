package activations

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Tanh(x *mat.Dense) *mat.Dense {
	res := mat.DenseCopyOf(x)
	res.Apply(func(i, j int, v float64) float64 {
		return math.Tanh(v)
	}, res)

	return res
}

func TanhPrime(x *mat.Dense) *mat.Dense {
	res := mat.DenseCopyOf(x)
	res.Apply(func(i, j int, v float64) float64 {
		// 1-tanh(x)^2
		return 1 - math.Pow(math.Tanh(v), 2)
	}, res)

	return res
}
