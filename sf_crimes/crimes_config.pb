language: PYTHON
name:     "crimes_job"

variable {
 name: "n_estimators"
 type: INT
 size: 1
 min:  1
 max:  150
}

variable {
 name: "criterion"
 type: ENUM
 size: 1
 options: "gini"
 options: "entropy"
}

variable {
 name: "max_depth"
 type: INT
 size: 1
 min:  0
 max:  40
}

variable {
 name: "min_samples_split"
 type: INT
 size: 1
 min:  2
 max:  15
}

variable {
 name: "min_samples_leaf"
 type: INT
 size: 1
 min:  1
 max:  10
}

variable {
 name: "max_features"
 type: ENUM
 size: 1
 options: "auto"
 options: "sqrt"
 options: "log2"
 options: ""
}

# variable {
#  name: "min_weight_fraction_leaf"
#  type: FLOAT
#  size: 1
#  min: 0
#  max: 0.3
# }

# Integer example
#
# variable {
#  name: "Y"
#  type: INT
#  size: 5
#  min:  -5
#  max:  5
# }

# Enumeration example
#
# variable {
#  name: "Z"
#  type: ENUM
#  size: 3
#  options: "foo"
#  options: "bar"
#  options: "baz"
# }
