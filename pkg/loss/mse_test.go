package loss

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMeanSquaredError(t *testing.T) {
	type args struct {
		yTrue *mat.Dense
		yPred *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{name: "Prediction is correct", args: args{yTrue: mat.NewDense(1, 1, []float64{1}), yPred: mat.NewDense(1, 1, []float64{1})}, want: 0},
		{name: "Prefiction is invalid", args: args{yTrue: mat.NewDense(1, 1, []float64{1}), yPred: mat.NewDense(1, 1, []float64{2})}, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MeanSquaredError(tt.args.yTrue, tt.args.yPred); got != tt.want {
				t.Errorf("MeanSquaredError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMeanSquaredErrorPrime(t *testing.T) {
	type args struct {
		yTrue *mat.Dense
		yPred *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want *mat.Dense
	}{
		{name: "Test 1", args: args{yTrue: mat.NewDense(1, 1, []float64{1}), yPred: mat.NewDense(1, 1, []float64{1})}, want: mat.NewDense(1, 1, []float64{0})},
		{name: "Test 2", args: args{yTrue: mat.NewDense(1, 1, []float64{1}), yPred: mat.NewDense(1, 1, []float64{2})}, want: mat.NewDense(1, 1, []float64{-2})},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MeanSquaredErrorPrime(tt.args.yTrue, tt.args.yPred); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MeanSquaredErrorPrime() = %v, want %v", got, tt.want)
			}
		})
	}
}
