package main

import (
	"fmt"

	"github.com/NiklasTreml/go-nn/pkg/activations"
	"github.com/NiklasTreml/go-nn/pkg/layers"
	"github.com/NiklasTreml/go-nn/pkg/loss"
	"github.com/NiklasTreml/go-nn/pkg/network"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// xor training data
	xTrain := []*mat.Dense{
		mat.NewDense(1, 2, []float64{0, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
	}
	yTrain := []*mat.Dense{
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
	}

	net := network.NewNeuralNet()
	net.Add(layers.NewDense(2, 3))
	net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))
	net.Add(layers.NewDense(3, 1))
	net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))

	net.Use(loss.MeanSquaredError, loss.MeanSquaredErrorPrime)

	net.Train(xTrain, yTrain, 1000, 0.1)

	out := net.Predict(xTrain)
	for i, x := range out {
		fmt.Printf("Input [%v, %v] -> Y_True: %v Y_Pred: %v\n", xTrain[i].At(0, 0), xTrain[i].At(0, 1), yTrain[i].At(0, 0), x.At(0, 0))
	}
}
