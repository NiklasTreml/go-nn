package activations

import (
	"math"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestTanh(t *testing.T) {
	type args struct {
		x *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want *mat.Dense
	}{
		{name: "It works", args: args{x: mat.NewDense(2, 2, []float64{0, 1, 2, 3})}, want: mat.NewDense(2, 2, []float64{math.Tanh(0), math.Tanh(1), math.Tanh(2), math.Tanh(3)})},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Tanh(tt.args.x); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Tanh() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTanhPrime(t *testing.T) {
	type args struct {
		x *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want *mat.Dense
	}{
		{name: "It works", args: args{x: mat.NewDense(2, 2, []float64{0, 1, 2, 3})}, want: mat.NewDense(2, 2, []float64{tanhPrime(0), tanhPrime(1), tanhPrime(2), tanhPrime(3)})},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := TanhPrime(tt.args.x); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("TanhPrime() = %v, want %v", got, tt.want)
			}
		})
	}
}

func tanhPrime(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}
