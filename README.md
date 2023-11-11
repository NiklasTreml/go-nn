# go-nn 🧠

go-nn is a neural network library written in Go, primarily developed for educational purposes. Please note that it's not yet stable or recommended for production use. Additionally, the library currently does not utilize GPU for training.

This library allows users to create and train neural networks for various machine learning tasks in a user-friendly, modular structure.

## Features 🚀

- **Modular Structure:** The library provides a modular structure for building neural networks, allowing users to easily add layers, define activation functions, and choose loss functions.
- **Example Usage:** An example showcasing how to use the library to train a neural network on the XOR dataset is provided in the `main` package.
- **Visualization:** The library includes functionalities to visualize training error over epochs.

## Installation ⚙️

To use go-nn in your Go project, simply run:

```sh
go get -u github.com/NiklasTreml/go-nn
```

## Usage 🤖

To create and train a neural network, follow these steps:

1. **Import the necessary packages:**

    ```go
    import (
        "fmt"
        "image/color"
        "github.com/NiklasTreml/go-nn/pkg/activations"
        "github.com/NiklasTreml/go-nn/pkg/layers"
        "github.com/NiklasTreml/go-nn/pkg/loss"
        "github.com/NiklasTreml/go-nn/pkg/network"
        "gonum.org/v1/gonum/mat"
        "gonum.org/v1/plot"
        "gonum.org/v1/plot/plotter"
        "gonum.org/v1/plot/vg"
    )
    ```

2. **Create training data:**

    ```go
    // XOR training data
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
    ```

3. **Build and train the neural network:**

    ```go
    net := network.NewNeuralNet()
    net.Add(layers.NewDense(2, 3))
    net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))
    net.Add(layers.NewDense(3, 1))
    net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))

    net.Use(loss.MeanSquaredError, loss.MeanSquaredErrorPrime)

    errorOverEpochs := net.Train(xTrain, yTrain, 500, 0.1)
    visualizeError(errorOverEpochs)

    out := net.Predict(xTrain)
    for i, x := range out {
        fmt.Printf("Input [%v, %v] -> Y_True: %v Y_Pred: %v\n", xTrain[i].At(0, 0), xTrain[i].At(0, 1), yTrain[i].At(0, 0), x.At(0, 0))
    }
    ```

4. **Predict using the trained network:**

    ```go
    // Get predictions for input data
    out := net.Predict(xTrain)
    // Display results
    // ...
    ```

For a more detailed understanding, refer to the example provided in the `main` package.

## Contribution 🤝

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/NiklasTreml/go-nn/issues) if you want to contribute.

## License 📝

This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.
