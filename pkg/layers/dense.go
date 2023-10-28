package layers

import (
	"github.com/NiklasTreml/go-nn/pkg/util"
	"gonum.org/v1/gonum/mat"
)

var _ Layer = (*Dense)(nil)

type Dense struct {
	weights *mat.Dense
	bias    *mat.Dense
	input   *mat.Dense
	output  *mat.Dense
}

func NewDense(inputSize int, outputSize int) *Dense {

	layer := &Dense{
		weights: mat.NewDense(inputSize, outputSize, util.RandomData(inputSize*outputSize)),
		bias:    mat.NewDense(1, outputSize, util.RandomData(outputSize)),
		input:   new(mat.Dense),
		output:  new(mat.Dense),
	}

	return layer
}

func (d *Dense) Forward(input *mat.Dense) *mat.Dense {
	d.input = mat.DenseCopyOf(input)

	output := new(mat.Dense)
	output.Mul(input, d.weights)
	output.Add(output, d.bias)
	d.output = mat.DenseCopyOf(output)

	return output
}

func (d *Dense) Backward(outputError *mat.Dense, alpha float64) *mat.Dense {
	inputError := new(mat.Dense)
	inputError.Mul(outputError, d.weights.T())

	weightsError := new(mat.Dense)
	weightsError.Mul(d.input.T(), outputError)

	weightsError.Scale(alpha, weightsError)
	d.weights.Sub(d.weights, weightsError)

	outputError.Scale(alpha, outputError)
	d.bias.Sub(d.bias, outputError)

	return inputError
}
