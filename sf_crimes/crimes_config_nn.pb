language: PYTHON
name:     "crimes_job_nn"

variable {
 name: "input_dropout"
 type: FLOAT
 size: 1
 min:  0
 max:  0.5
}

variable {
 name: "layers"
 type: INT
 size: 1
 min:  1
 max:  3
}

# variable {
#  name: "hidden_func"
#  type: ENUM
#  size: 1
#  options: "Rectifier"
#  options: "Tanh"
#  options: "Sigmoid"
# }

variable {
 name: "hidden_units"
 type: ENUM
 size: 1
 options: "64"
 options: "128"
 options: "256"
}

variable {
 name: "hidden_dropout"
 type: FLOAT
 size: 1
 min:  0
 max:  0.75
}


variable {
 name: "learning_rate"
 type: FLOAT
 size: 1
 min:  0.01
 max:  0.1
}

variable {
 name: "weight_decay"
 type: FLOAT
 size: 1
 min:  0
 max:  0.01
}
