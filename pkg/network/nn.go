package network

import (
	"fmt"

	"github.com/NiklasTreml/go-nn/pkg/layers"
	"github.com/NiklasTreml/go-nn/pkg/loss"
	"gonum.org/v1/gonum/mat"
)

type NeuralNet struct {
	layers    []layers.Layer
	loss      loss.LossFn
	lossPrime loss.LossPrimeFn
}

func NewNeuralNet() *NeuralNet {
	return &NeuralNet{
		layers: []layers.Layer{},
	}
}

func (nn *NeuralNet) Add(layer layers.Layer) {
	nn.layers = append(nn.layers, layer)
}

func (nn *NeuralNet) Use(loss loss.LossFn, lossPrime loss.LossPrimeFn) {
	nn.loss = loss
	nn.lossPrime = lossPrime
}

func (nn *NeuralNet) Predict(inputs []*mat.Dense) []*mat.Dense {
	results := []*mat.Dense{}

	for _, input := range inputs {
		output := mat.DenseCopyOf(input)
		for _, layer := range nn.layers {

			output = layer.Forward(output)
		}
		results = append(results, output)
	}

	return results
}

func (nn *NeuralNet) Train(xTrain, yTrain []*mat.Dense, epochs int, alpha float64) []float64 {
	errorOverEpochs := []float64{}
	for epoch := 0; epoch < epochs; epoch++ {
		displayError := 0.0

		for iX, input := range xTrain {
			output := input
			for _, layer := range nn.layers {
				output = layer.Forward(output)
			}

			displayError += nn.loss(output, yTrain[iX])

			// backprop
			errorPrime := nn.lossPrime(output, yTrain[iX])

			for i := len(nn.layers) - 1; i >= 0; i-- {
				errorPrime = nn.layers[i].Backward(errorPrime, alpha)
			}
		}

		displayError /= float64(len(xTrain))
		fmt.Printf("Epoch %d/%d Error=%f\r", epoch+1, epochs, displayError)
		errorOverEpochs = append(errorOverEpochs, displayError)
	}
	fmt.Printf("\n")

	return errorOverEpochs

}
