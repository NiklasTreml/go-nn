package loss

import (
	"gonum.org/v1/gonum/mat"
)

func MeanSquaredError(yTrue, yPred *mat.Dense) float64 {
	res := mat.DenseCopyOf(yTrue)

	res.Sub(res, yPred)
	res.MulElem(res, res)

	sum := mat.Sum(res)

	mse := sum / float64(yPred.RawMatrix().Rows)

	return mse
}
func MeanSquaredErrorPrime(yTrue, yPred *mat.Dense) *mat.Dense {
	msePrime := new(mat.Dense)

	msePrime.Sub(yTrue, yPred)

	msePrime.Scale(2.0/float64(yPred.RawMatrix().Rows), msePrime)

	return msePrime
}
