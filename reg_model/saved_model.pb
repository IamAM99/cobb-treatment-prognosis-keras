��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-0-gc256c071bb28��
x
batch_norm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatch_norm/gamma
q
$batch_norm/gamma/Read/ReadVariableOpReadVariableOpbatch_norm/gamma*
_output_shapes
:*
dtype0
v
batch_norm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebatch_norm/beta
o
#batch_norm/beta/Read/ReadVariableOpReadVariableOpbatch_norm/beta*
_output_shapes
:*
dtype0
�
batch_norm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebatch_norm/moving_mean
}
*batch_norm/moving_mean/Read/ReadVariableOpReadVariableOpbatch_norm/moving_mean*
_output_shapes
:*
dtype0
�
batch_norm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_norm/moving_variance
�
.batch_norm/moving_variance/Read/ReadVariableOpReadVariableOpbatch_norm/moving_variance*
_output_shapes
:*
dtype0
{
hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namehidden_1/kernel
t
#hidden_1/kernel/Read/ReadVariableOpReadVariableOphidden_1/kernel*
_output_shapes
:	�*
dtype0
s
hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namehidden_1/bias
l
!hidden_1/bias/Read/ReadVariableOpReadVariableOphidden_1/bias*
_output_shapes	
:�*
dtype0
|
hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namehidden_2/kernel
u
#hidden_2/kernel/Read/ReadVariableOpReadVariableOphidden_2/kernel* 
_output_shapes
:
��*
dtype0
s
hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namehidden_2/bias
l
!hidden_2/bias/Read/ReadVariableOpReadVariableOphidden_2/bias*
_output_shapes	
:�*
dtype0
|
hidden_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namehidden_3/kernel
u
#hidden_3/kernel/Read/ReadVariableOpReadVariableOphidden_3/kernel* 
_output_shapes
:
��*
dtype0
s
hidden_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namehidden_3/bias
l
!hidden_3/bias/Read/ReadVariableOpReadVariableOphidden_3/bias*
_output_shapes	
:�*
dtype0
|
hidden_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namehidden_4/kernel
u
#hidden_4/kernel/Read/ReadVariableOpReadVariableOphidden_4/kernel* 
_output_shapes
:
��*
dtype0
s
hidden_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namehidden_4/bias
l
!hidden_4/bias/Read/ReadVariableOpReadVariableOphidden_4/bias*
_output_shapes	
:�*
dtype0
�
output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameoutput_layer/kernel
|
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes
:	�*
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
n
squared_sumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesquared_sum
g
squared_sum/Read/ReadVariableOpReadVariableOpsquared_sum*
_output_shapes
:*
dtype0
^
sumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesum
W
sum/Read/ReadVariableOpReadVariableOpsum*
_output_shapes
:*
dtype0
h
residualVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
residual
a
residual/Read/ReadVariableOpReadVariableOpresidual*
_output_shapes
:*
dtype0
f
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	count_2
_
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
:*
dtype0
j
num_samplesVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namenum_samples
c
num_samples/Read/ReadVariableOpReadVariableOpnum_samples*
_output_shapes
: *
dtype0
�
Adam/batch_norm/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batch_norm/gamma/m

+Adam/batch_norm/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batch_norm/gamma/m*
_output_shapes
:*
dtype0
�
Adam/batch_norm/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/batch_norm/beta/m
}
*Adam/batch_norm/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_norm/beta/m*
_output_shapes
:*
dtype0
�
Adam/hidden_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/hidden_1/kernel/m
�
*Adam/hidden_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/hidden_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/hidden_1/bias/m
z
(Adam/hidden_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/hidden_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/hidden_2/kernel/m
�
*Adam/hidden_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_2/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/hidden_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/hidden_2/bias/m
z
(Adam/hidden_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/hidden_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/hidden_3/kernel/m
�
*Adam/hidden_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_3/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/hidden_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/hidden_3/bias/m
z
(Adam/hidden_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/hidden_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/hidden_4/kernel/m
�
*Adam/hidden_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_4/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/hidden_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/hidden_4/bias/m
z
(Adam/hidden_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_4/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameAdam/output_layer/kernel/m
�
.Adam/output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/m
�
,Adam/output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/m*
_output_shapes
:*
dtype0
�
Adam/batch_norm/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batch_norm/gamma/v

+Adam/batch_norm/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batch_norm/gamma/v*
_output_shapes
:*
dtype0
�
Adam/batch_norm/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/batch_norm/beta/v
}
*Adam/batch_norm/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_norm/beta/v*
_output_shapes
:*
dtype0
�
Adam/hidden_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/hidden_1/kernel/v
�
*Adam/hidden_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/hidden_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/hidden_1/bias/v
z
(Adam/hidden_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/hidden_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/hidden_2/kernel/v
�
*Adam/hidden_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_2/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/hidden_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/hidden_2/bias/v
z
(Adam/hidden_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/hidden_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/hidden_3/kernel/v
�
*Adam/hidden_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_3/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/hidden_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/hidden_3/bias/v
z
(Adam/hidden_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/hidden_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/hidden_4/kernel/v
�
*Adam/hidden_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_4/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/hidden_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/hidden_4/bias/v
z
(Adam/hidden_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_4/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameAdam/output_layer/kernel/v
�
.Adam/output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/v
�
,Adam/output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�F
value�FB�F B�F
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
�

4beta_1

5beta_2
	6decay
7learning_rate
8itermnmompmqmrms"mt#mu(mv)mw.mx/myvzv{v|v}v~v"v�#v�(v�)v�.v�/v�
f
0
1
2
3
4
5
6
7
"8
#9
(10
)11
.12
/13
V
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
 
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
	trainable_variables

regularization_losses
 
 
[Y
VARIABLE_VALUEbatch_norm/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_norm/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_norm/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbatch_norm/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEhidden_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEhidden_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
 regularization_losses
[Y
VARIABLE_VALUEhidden_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
[Y
VARIABLE_VALUEhidden_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
*	variables
+trainable_variables
,regularization_losses
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
0	variables
1trainable_variables
2regularization_losses
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE

0
1
*
0
1
2
3
4
5

\0
]1
^2
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	_total
	`count
a	variables
b	keras_api
4
	ctotal
	dcount
e	variables
f	keras_api
k
gsquared_sum
hsum
iresidual
ires
	jcount
knum_samples
l	variables
m	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

a	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

e	variables
[Y
VARIABLE_VALUEsquared_sum:keras_api/metrics/2/squared_sum/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEsum2keras_api/metrics/2/sum/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEresidual7keras_api/metrics/2/residual/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEnum_samples:keras_api/metrics/2/num_samples/.ATTRIBUTES/VARIABLE_VALUE
#
g0
h1
i2
j3
k4

l	variables
~|
VARIABLE_VALUEAdam/batch_norm/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/batch_norm/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/output_layer/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output_layer/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batch_norm/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/batch_norm/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/output_layer/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output_layer/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_input_layerPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerbatch_norm/moving_variancebatch_norm/gammabatch_norm/moving_meanbatch_norm/betahidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biashidden_4/kernelhidden_4/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_343656
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$batch_norm/gamma/Read/ReadVariableOp#batch_norm/beta/Read/ReadVariableOp*batch_norm/moving_mean/Read/ReadVariableOp.batch_norm/moving_variance/Read/ReadVariableOp#hidden_1/kernel/Read/ReadVariableOp!hidden_1/bias/Read/ReadVariableOp#hidden_2/kernel/Read/ReadVariableOp!hidden_2/bias/Read/ReadVariableOp#hidden_3/kernel/Read/ReadVariableOp!hidden_3/bias/Read/ReadVariableOp#hidden_4/kernel/Read/ReadVariableOp!hidden_4/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpsquared_sum/Read/ReadVariableOpsum/Read/ReadVariableOpresidual/Read/ReadVariableOpcount_2/Read/ReadVariableOpnum_samples/Read/ReadVariableOp+Adam/batch_norm/gamma/m/Read/ReadVariableOp*Adam/batch_norm/beta/m/Read/ReadVariableOp*Adam/hidden_1/kernel/m/Read/ReadVariableOp(Adam/hidden_1/bias/m/Read/ReadVariableOp*Adam/hidden_2/kernel/m/Read/ReadVariableOp(Adam/hidden_2/bias/m/Read/ReadVariableOp*Adam/hidden_3/kernel/m/Read/ReadVariableOp(Adam/hidden_3/bias/m/Read/ReadVariableOp*Adam/hidden_4/kernel/m/Read/ReadVariableOp(Adam/hidden_4/bias/m/Read/ReadVariableOp.Adam/output_layer/kernel/m/Read/ReadVariableOp,Adam/output_layer/bias/m/Read/ReadVariableOp+Adam/batch_norm/gamma/v/Read/ReadVariableOp*Adam/batch_norm/beta/v/Read/ReadVariableOp*Adam/hidden_1/kernel/v/Read/ReadVariableOp(Adam/hidden_1/bias/v/Read/ReadVariableOp*Adam/hidden_2/kernel/v/Read/ReadVariableOp(Adam/hidden_2/bias/v/Read/ReadVariableOp*Adam/hidden_3/kernel/v/Read/ReadVariableOp(Adam/hidden_3/bias/v/Read/ReadVariableOp*Adam/hidden_4/kernel/v/Read/ReadVariableOp(Adam/hidden_4/bias/v/Read/ReadVariableOp.Adam/output_layer/kernel/v/Read/ReadVariableOp,Adam/output_layer/bias/v/Read/ReadVariableOpConst*A
Tin:
826	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_344342
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_norm/gammabatch_norm/betabatch_norm/moving_meanbatch_norm/moving_variancehidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biashidden_4/kernelhidden_4/biasoutput_layer/kerneloutput_layer/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1squared_sumsumresidualcount_2num_samplesAdam/batch_norm/gamma/mAdam/batch_norm/beta/mAdam/hidden_1/kernel/mAdam/hidden_1/bias/mAdam/hidden_2/kernel/mAdam/hidden_2/bias/mAdam/hidden_3/kernel/mAdam/hidden_3/bias/mAdam/hidden_4/kernel/mAdam/hidden_4/bias/mAdam/output_layer/kernel/mAdam/output_layer/bias/mAdam/batch_norm/gamma/vAdam/batch_norm/beta/vAdam/hidden_1/kernel/vAdam/hidden_1/bias/vAdam/hidden_2/kernel/vAdam/hidden_2/bias/vAdam/hidden_3/kernel/vAdam/hidden_3/bias/vAdam/hidden_4/kernel/vAdam/hidden_4/bias/vAdam/output_layer/kernel/vAdam/output_layer/bias/v*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_344508֕

�
�
F__inference_batch_norm_layer_call_and_return_conditional_losses_343018

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_344130M
:hidden_1_kernel_regularizer_square_readvariableop_resource:	�
identity��1hidden_1/kernel/Regularizer/Square/ReadVariableOp�
1hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:hidden_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"hidden_1/kernel/Regularizer/SquareSquare9hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�r
!hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_1/kernel/Regularizer/SumSum&hidden_1/kernel/Regularizer/Square:y:0*hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_1/kernel/Regularizer/mulMul*hidden_1/kernel/Regularizer/mul/x:output:0(hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#hidden_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^hidden_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1hidden_1/kernel/Regularizer/Square/ReadVariableOp1hidden_1/kernel/Regularizer/Square/ReadVariableOp
�w
�
C__inference_reg_net_layer_call_and_return_conditional_losses_343892

inputs@
2batch_norm_assignmovingavg_readvariableop_resource:B
4batch_norm_assignmovingavg_1_readvariableop_resource:>
0batch_norm_batchnorm_mul_readvariableop_resource::
,batch_norm_batchnorm_readvariableop_resource::
'hidden_1_matmul_readvariableop_resource:	�7
(hidden_1_biasadd_readvariableop_resource:	�;
'hidden_2_matmul_readvariableop_resource:
��7
(hidden_2_biasadd_readvariableop_resource:	�;
'hidden_3_matmul_readvariableop_resource:
��7
(hidden_3_biasadd_readvariableop_resource:	�;
'hidden_4_matmul_readvariableop_resource:
��7
(hidden_4_biasadd_readvariableop_resource:	�>
+output_layer_matmul_readvariableop_resource:	�:
,output_layer_biasadd_readvariableop_resource:
identity��batch_norm/AssignMovingAvg�)batch_norm/AssignMovingAvg/ReadVariableOp�batch_norm/AssignMovingAvg_1�+batch_norm/AssignMovingAvg_1/ReadVariableOp�#batch_norm/batchnorm/ReadVariableOp�'batch_norm/batchnorm/mul/ReadVariableOp�hidden_1/BiasAdd/ReadVariableOp�hidden_1/MatMul/ReadVariableOp�1hidden_1/kernel/Regularizer/Square/ReadVariableOp�hidden_2/BiasAdd/ReadVariableOp�hidden_2/MatMul/ReadVariableOp�1hidden_2/kernel/Regularizer/Square/ReadVariableOp�hidden_3/BiasAdd/ReadVariableOp�hidden_3/MatMul/ReadVariableOp�1hidden_3/kernel/Regularizer/Square/ReadVariableOp�hidden_4/BiasAdd/ReadVariableOp�hidden_4/MatMul/ReadVariableOp�1hidden_4/kernel/Regularizer/Square/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOps
)batch_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
batch_norm/moments/meanMeaninputs2batch_norm/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(z
batch_norm/moments/StopGradientStopGradient batch_norm/moments/mean:output:0*
T0*
_output_shapes

:�
$batch_norm/moments/SquaredDifferenceSquaredDifferenceinputs(batch_norm/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������w
-batch_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
batch_norm/moments/varianceMean(batch_norm/moments/SquaredDifference:z:06batch_norm/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
batch_norm/moments/SqueezeSqueeze batch_norm/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
batch_norm/moments/Squeeze_1Squeeze$batch_norm/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 e
 batch_norm/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
)batch_norm/AssignMovingAvg/ReadVariableOpReadVariableOp2batch_norm_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
batch_norm/AssignMovingAvg/subSub1batch_norm/AssignMovingAvg/ReadVariableOp:value:0#batch_norm/moments/Squeeze:output:0*
T0*
_output_shapes
:�
batch_norm/AssignMovingAvg/mulMul"batch_norm/AssignMovingAvg/sub:z:0)batch_norm/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
batch_norm/AssignMovingAvgAssignSubVariableOp2batch_norm_assignmovingavg_readvariableop_resource"batch_norm/AssignMovingAvg/mul:z:0*^batch_norm/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0g
"batch_norm/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
+batch_norm/AssignMovingAvg_1/ReadVariableOpReadVariableOp4batch_norm_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
 batch_norm/AssignMovingAvg_1/subSub3batch_norm/AssignMovingAvg_1/ReadVariableOp:value:0%batch_norm/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
 batch_norm/AssignMovingAvg_1/mulMul$batch_norm/AssignMovingAvg_1/sub:z:0+batch_norm/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
batch_norm/AssignMovingAvg_1AssignSubVariableOp4batch_norm_assignmovingavg_1_readvariableop_resource$batch_norm/AssignMovingAvg_1/mul:z:0,^batch_norm/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0_
batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_norm/batchnorm/addAddV2%batch_norm/moments/Squeeze_1:output:0#batch_norm/batchnorm/add/y:output:0*
T0*
_output_shapes
:f
batch_norm/batchnorm/RsqrtRsqrtbatch_norm/batchnorm/add:z:0*
T0*
_output_shapes
:�
'batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp0batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batch_norm/batchnorm/mulMulbatch_norm/batchnorm/Rsqrt:y:0/batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:y
batch_norm/batchnorm/mul_1Mulinputsbatch_norm/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
batch_norm/batchnorm/mul_2Mul#batch_norm/moments/Squeeze:output:0batch_norm/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_norm/batchnorm/ReadVariableOpReadVariableOp,batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batch_norm/batchnorm/subSub+batch_norm/batchnorm/ReadVariableOp:value:0batch_norm/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
batch_norm/batchnorm/add_1AddV2batch_norm/batchnorm/mul_1:z:0batch_norm/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
hidden_1/MatMulMatMulbatch_norm/batchnorm/add_1:z:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hidden_2/MatMulMatMulhidden_1/Relu:activations:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hidden_3/MatMulMatMulhidden_2/Relu:activations:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
hidden_3/ReluReluhidden_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
hidden_4/MatMul/ReadVariableOpReadVariableOp'hidden_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hidden_4/MatMulMatMulhidden_3/Relu:activations:0&hidden_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
hidden_4/BiasAdd/ReadVariableOpReadVariableOp(hidden_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
hidden_4/BiasAddBiasAddhidden_4/MatMul:product:0'hidden_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
hidden_4/ReluReluhidden_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
output_layer/MatMulMatMulhidden_4/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"hidden_1/kernel/Regularizer/SquareSquare9hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�r
!hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_1/kernel/Regularizer/SumSum&hidden_1/kernel/Regularizer/Square:y:0*hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_1/kernel/Regularizer/mulMul*hidden_1/kernel/Regularizer/mul/x:output:0(hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_2/kernel/Regularizer/SquareSquare9hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_2/kernel/Regularizer/SumSum&hidden_2/kernel/Regularizer/Square:y:0*hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_2/kernel/Regularizer/mulMul*hidden_2/kernel/Regularizer/mul/x:output:0(hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_3/kernel/Regularizer/SquareSquare9hidden_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_3/kernel/Regularizer/SumSum&hidden_3/kernel/Regularizer/Square:y:0*hidden_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_3/kernel/Regularizer/mulMul*hidden_3/kernel/Regularizer/mul/x:output:0(hidden_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'hidden_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_4/kernel/Regularizer/SquareSquare9hidden_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_4/kernel/Regularizer/SumSum&hidden_4/kernel/Regularizer/Square:y:0*hidden_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_4/kernel/Regularizer/mulMul*hidden_4/kernel/Regularizer/mul/x:output:0(hidden_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: l
IdentityIdentityoutput_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batch_norm/AssignMovingAvg*^batch_norm/AssignMovingAvg/ReadVariableOp^batch_norm/AssignMovingAvg_1,^batch_norm/AssignMovingAvg_1/ReadVariableOp$^batch_norm/batchnorm/ReadVariableOp(^batch_norm/batchnorm/mul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp2^hidden_1/kernel/Regularizer/Square/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp2^hidden_2/kernel/Regularizer/Square/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp2^hidden_3/kernel/Regularizer/Square/ReadVariableOp ^hidden_4/BiasAdd/ReadVariableOp^hidden_4/MatMul/ReadVariableOp2^hidden_4/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 28
batch_norm/AssignMovingAvgbatch_norm/AssignMovingAvg2V
)batch_norm/AssignMovingAvg/ReadVariableOp)batch_norm/AssignMovingAvg/ReadVariableOp2<
batch_norm/AssignMovingAvg_1batch_norm/AssignMovingAvg_12Z
+batch_norm/AssignMovingAvg_1/ReadVariableOp+batch_norm/AssignMovingAvg_1/ReadVariableOp2J
#batch_norm/batchnorm/ReadVariableOp#batch_norm/batchnorm/ReadVariableOp2R
'batch_norm/batchnorm/mul/ReadVariableOp'batch_norm/batchnorm/mul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2f
1hidden_1/kernel/Regularizer/Square/ReadVariableOp1hidden_1/kernel/Regularizer/Square/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2f
1hidden_2/kernel/Regularizer/Square/ReadVariableOp1hidden_2/kernel/Regularizer/Square/ReadVariableOp2B
hidden_3/BiasAdd/ReadVariableOphidden_3/BiasAdd/ReadVariableOp2@
hidden_3/MatMul/ReadVariableOphidden_3/MatMul/ReadVariableOp2f
1hidden_3/kernel/Regularizer/Square/ReadVariableOp1hidden_3/kernel/Regularizer/Square/ReadVariableOp2B
hidden_4/BiasAdd/ReadVariableOphidden_4/BiasAdd/ReadVariableOp2@
hidden_4/MatMul/ReadVariableOphidden_4/MatMul/ReadVariableOp2f
1hidden_4/kernel/Regularizer/Square/ReadVariableOp1hidden_4/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_hidden_1_layer_call_fn_343987

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_343109p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_344508
file_prefix/
!assignvariableop_batch_norm_gamma:0
"assignvariableop_1_batch_norm_beta:7
)assignvariableop_2_batch_norm_moving_mean:;
-assignvariableop_3_batch_norm_moving_variance:5
"assignvariableop_4_hidden_1_kernel:	�/
 assignvariableop_5_hidden_1_bias:	�6
"assignvariableop_6_hidden_2_kernel:
��/
 assignvariableop_7_hidden_2_bias:	�6
"assignvariableop_8_hidden_3_kernel:
��/
 assignvariableop_9_hidden_3_bias:	�7
#assignvariableop_10_hidden_4_kernel:
��0
!assignvariableop_11_hidden_4_bias:	�:
'assignvariableop_12_output_layer_kernel:	�3
%assignvariableop_13_output_layer_bias:$
assignvariableop_14_beta_1: $
assignvariableop_15_beta_2: #
assignvariableop_16_decay: +
!assignvariableop_17_learning_rate: '
assignvariableop_18_adam_iter:	 #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: -
assignvariableop_23_squared_sum:%
assignvariableop_24_sum:*
assignvariableop_25_residual:)
assignvariableop_26_count_2:)
assignvariableop_27_num_samples: 9
+assignvariableop_28_adam_batch_norm_gamma_m:8
*assignvariableop_29_adam_batch_norm_beta_m:=
*assignvariableop_30_adam_hidden_1_kernel_m:	�7
(assignvariableop_31_adam_hidden_1_bias_m:	�>
*assignvariableop_32_adam_hidden_2_kernel_m:
��7
(assignvariableop_33_adam_hidden_2_bias_m:	�>
*assignvariableop_34_adam_hidden_3_kernel_m:
��7
(assignvariableop_35_adam_hidden_3_bias_m:	�>
*assignvariableop_36_adam_hidden_4_kernel_m:
��7
(assignvariableop_37_adam_hidden_4_bias_m:	�A
.assignvariableop_38_adam_output_layer_kernel_m:	�:
,assignvariableop_39_adam_output_layer_bias_m:9
+assignvariableop_40_adam_batch_norm_gamma_v:8
*assignvariableop_41_adam_batch_norm_beta_v:=
*assignvariableop_42_adam_hidden_1_kernel_v:	�7
(assignvariableop_43_adam_hidden_1_bias_v:	�>
*assignvariableop_44_adam_hidden_2_kernel_v:
��7
(assignvariableop_45_adam_hidden_2_bias_v:	�>
*assignvariableop_46_adam_hidden_3_kernel_v:
��7
(assignvariableop_47_adam_hidden_3_bias_v:	�>
*assignvariableop_48_adam_hidden_4_kernel_v:
��7
(assignvariableop_49_adam_hidden_4_bias_v:	�A
.assignvariableop_50_adam_output_layer_kernel_v:	�:
,assignvariableop_51_adam_output_layer_bias_v:
identity_53��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/squared_sum/.ATTRIBUTES/VARIABLE_VALUEB2keras_api/metrics/2/sum/.ATTRIBUTES/VARIABLE_VALUEB7keras_api/metrics/2/residual/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/num_samples/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_batch_norm_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_batch_norm_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp)assignvariableop_2_batch_norm_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_norm_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_hidden_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_hidden_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_hidden_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_hidden_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_hidden_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_hidden_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_hidden_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_hidden_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_output_layer_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp%assignvariableop_13_output_layer_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_beta_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_beta_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_decayIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_squared_sumIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_sumIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_residualIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_num_samplesIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_batch_norm_gamma_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_batch_norm_beta_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_hidden_1_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_hidden_1_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_hidden_2_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_hidden_2_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_hidden_3_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_hidden_3_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_hidden_4_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_hidden_4_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp.assignvariableop_38_adam_output_layer_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_output_layer_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_batch_norm_gamma_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_batch_norm_beta_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_hidden_1_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_hidden_1_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_hidden_2_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_hidden_2_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_hidden_3_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_hidden_3_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_hidden_4_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_hidden_4_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp.assignvariableop_50_adam_output_layer_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_output_layer_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_53Identity_53:output:0*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
$__inference_signature_wrapper_343656
input_layer
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_342994o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
__inference_loss_fn_2_344152N
:hidden_3_kernel_regularizer_square_readvariableop_resource:
��
identity��1hidden_3/kernel/Regularizer/Square/ReadVariableOp�
1hidden_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:hidden_3_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_3/kernel/Regularizer/SquareSquare9hidden_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_3/kernel/Regularizer/SumSum&hidden_3/kernel/Regularizer/Square:y:0*hidden_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_3/kernel/Regularizer/mulMul*hidden_3/kernel/Regularizer/mul/x:output:0(hidden_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#hidden_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^hidden_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1hidden_3/kernel/Regularizer/Square/ReadVariableOp1hidden_3/kernel/Regularizer/Square/ReadVariableOp
�B
�
C__inference_reg_net_layer_call_and_return_conditional_losses_343529
input_layer
batch_norm_343470:
batch_norm_343472:
batch_norm_343474:
batch_norm_343476:"
hidden_1_343479:	�
hidden_1_343481:	�#
hidden_2_343484:
��
hidden_2_343486:	�#
hidden_3_343489:
��
hidden_3_343491:	�#
hidden_4_343494:
��
hidden_4_343496:	�&
output_layer_343499:	�!
output_layer_343501:
identity��"batch_norm/StatefulPartitionedCall� hidden_1/StatefulPartitionedCall�1hidden_1/kernel/Regularizer/Square/ReadVariableOp� hidden_2/StatefulPartitionedCall�1hidden_2/kernel/Regularizer/Square/ReadVariableOp� hidden_3/StatefulPartitionedCall�1hidden_3/kernel/Regularizer/Square/ReadVariableOp� hidden_4/StatefulPartitionedCall�1hidden_4/kernel/Regularizer/Square/ReadVariableOp�$output_layer/StatefulPartitionedCall�
"batch_norm/StatefulPartitionedCallStatefulPartitionedCallinput_layerbatch_norm_343470batch_norm_343472batch_norm_343474batch_norm_343476*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_343018�
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0hidden_1_343479hidden_1_343481*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_343109�
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_343484hidden_2_343486*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_343132�
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_343489hidden_3_343491*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_3_layer_call_and_return_conditional_losses_343155�
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_343494hidden_4_343496*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_4_layer_call_and_return_conditional_losses_343178�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0output_layer_343499output_layer_343501*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_343194�
1hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_1_343479*
_output_shapes
:	�*
dtype0�
"hidden_1/kernel/Regularizer/SquareSquare9hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�r
!hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_1/kernel/Regularizer/SumSum&hidden_1/kernel/Regularizer/Square:y:0*hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_1/kernel/Regularizer/mulMul*hidden_1/kernel/Regularizer/mul/x:output:0(hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_2_343484* 
_output_shapes
:
��*
dtype0�
"hidden_2/kernel/Regularizer/SquareSquare9hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_2/kernel/Regularizer/SumSum&hidden_2/kernel/Regularizer/Square:y:0*hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_2/kernel/Regularizer/mulMul*hidden_2/kernel/Regularizer/mul/x:output:0(hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_3_343489* 
_output_shapes
:
��*
dtype0�
"hidden_3/kernel/Regularizer/SquareSquare9hidden_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_3/kernel/Regularizer/SumSum&hidden_3/kernel/Regularizer/Square:y:0*hidden_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_3/kernel/Regularizer/mulMul*hidden_3/kernel/Regularizer/mul/x:output:0(hidden_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_4_343494* 
_output_shapes
:
��*
dtype0�
"hidden_4/kernel/Regularizer/SquareSquare9hidden_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_4/kernel/Regularizer/SumSum&hidden_4/kernel/Regularizer/Square:y:0*hidden_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_4/kernel/Regularizer/mulMul*hidden_4/kernel/Regularizer/mul/x:output:0(hidden_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^batch_norm/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall2^hidden_1/kernel/Regularizer/Square/ReadVariableOp!^hidden_2/StatefulPartitionedCall2^hidden_2/kernel/Regularizer/Square/ReadVariableOp!^hidden_3/StatefulPartitionedCall2^hidden_3/kernel/Regularizer/Square/ReadVariableOp!^hidden_4/StatefulPartitionedCall2^hidden_4/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2f
1hidden_1/kernel/Regularizer/Square/ReadVariableOp1hidden_1/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2f
1hidden_2/kernel/Regularizer/Square/ReadVariableOp1hidden_2/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2f
1hidden_3/kernel/Regularizer/Square/ReadVariableOp1hidden_3/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2f
1hidden_4/kernel/Regularizer/Square/ReadVariableOp1hidden_4/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�%
�
F__inference_batch_norm_layer_call_and_return_conditional_losses_343972

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_reg_net_layer_call_fn_343689

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reg_net_layer_call_and_return_conditional_losses_343225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�
!__inference__wrapped_model_342994
input_layerB
4reg_net_batch_norm_batchnorm_readvariableop_resource:F
8reg_net_batch_norm_batchnorm_mul_readvariableop_resource:D
6reg_net_batch_norm_batchnorm_readvariableop_1_resource:D
6reg_net_batch_norm_batchnorm_readvariableop_2_resource:B
/reg_net_hidden_1_matmul_readvariableop_resource:	�?
0reg_net_hidden_1_biasadd_readvariableop_resource:	�C
/reg_net_hidden_2_matmul_readvariableop_resource:
��?
0reg_net_hidden_2_biasadd_readvariableop_resource:	�C
/reg_net_hidden_3_matmul_readvariableop_resource:
��?
0reg_net_hidden_3_biasadd_readvariableop_resource:	�C
/reg_net_hidden_4_matmul_readvariableop_resource:
��?
0reg_net_hidden_4_biasadd_readvariableop_resource:	�F
3reg_net_output_layer_matmul_readvariableop_resource:	�B
4reg_net_output_layer_biasadd_readvariableop_resource:
identity��+reg_net/batch_norm/batchnorm/ReadVariableOp�-reg_net/batch_norm/batchnorm/ReadVariableOp_1�-reg_net/batch_norm/batchnorm/ReadVariableOp_2�/reg_net/batch_norm/batchnorm/mul/ReadVariableOp�'reg_net/hidden_1/BiasAdd/ReadVariableOp�&reg_net/hidden_1/MatMul/ReadVariableOp�'reg_net/hidden_2/BiasAdd/ReadVariableOp�&reg_net/hidden_2/MatMul/ReadVariableOp�'reg_net/hidden_3/BiasAdd/ReadVariableOp�&reg_net/hidden_3/MatMul/ReadVariableOp�'reg_net/hidden_4/BiasAdd/ReadVariableOp�&reg_net/hidden_4/MatMul/ReadVariableOp�+reg_net/output_layer/BiasAdd/ReadVariableOp�*reg_net/output_layer/MatMul/ReadVariableOp�
+reg_net/batch_norm/batchnorm/ReadVariableOpReadVariableOp4reg_net_batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0g
"reg_net/batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 reg_net/batch_norm/batchnorm/addAddV23reg_net/batch_norm/batchnorm/ReadVariableOp:value:0+reg_net/batch_norm/batchnorm/add/y:output:0*
T0*
_output_shapes
:v
"reg_net/batch_norm/batchnorm/RsqrtRsqrt$reg_net/batch_norm/batchnorm/add:z:0*
T0*
_output_shapes
:�
/reg_net/batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp8reg_net_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
 reg_net/batch_norm/batchnorm/mulMul&reg_net/batch_norm/batchnorm/Rsqrt:y:07reg_net/batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
"reg_net/batch_norm/batchnorm/mul_1Mulinput_layer$reg_net/batch_norm/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
-reg_net/batch_norm/batchnorm/ReadVariableOp_1ReadVariableOp6reg_net_batch_norm_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
"reg_net/batch_norm/batchnorm/mul_2Mul5reg_net/batch_norm/batchnorm/ReadVariableOp_1:value:0$reg_net/batch_norm/batchnorm/mul:z:0*
T0*
_output_shapes
:�
-reg_net/batch_norm/batchnorm/ReadVariableOp_2ReadVariableOp6reg_net_batch_norm_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
 reg_net/batch_norm/batchnorm/subSub5reg_net/batch_norm/batchnorm/ReadVariableOp_2:value:0&reg_net/batch_norm/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
"reg_net/batch_norm/batchnorm/add_1AddV2&reg_net/batch_norm/batchnorm/mul_1:z:0$reg_net/batch_norm/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
&reg_net/hidden_1/MatMul/ReadVariableOpReadVariableOp/reg_net_hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
reg_net/hidden_1/MatMulMatMul&reg_net/batch_norm/batchnorm/add_1:z:0.reg_net/hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'reg_net/hidden_1/BiasAdd/ReadVariableOpReadVariableOp0reg_net_hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
reg_net/hidden_1/BiasAddBiasAdd!reg_net/hidden_1/MatMul:product:0/reg_net/hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
reg_net/hidden_1/ReluRelu!reg_net/hidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&reg_net/hidden_2/MatMul/ReadVariableOpReadVariableOp/reg_net_hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
reg_net/hidden_2/MatMulMatMul#reg_net/hidden_1/Relu:activations:0.reg_net/hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'reg_net/hidden_2/BiasAdd/ReadVariableOpReadVariableOp0reg_net_hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
reg_net/hidden_2/BiasAddBiasAdd!reg_net/hidden_2/MatMul:product:0/reg_net/hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
reg_net/hidden_2/ReluRelu!reg_net/hidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&reg_net/hidden_3/MatMul/ReadVariableOpReadVariableOp/reg_net_hidden_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
reg_net/hidden_3/MatMulMatMul#reg_net/hidden_2/Relu:activations:0.reg_net/hidden_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'reg_net/hidden_3/BiasAdd/ReadVariableOpReadVariableOp0reg_net_hidden_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
reg_net/hidden_3/BiasAddBiasAdd!reg_net/hidden_3/MatMul:product:0/reg_net/hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
reg_net/hidden_3/ReluRelu!reg_net/hidden_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&reg_net/hidden_4/MatMul/ReadVariableOpReadVariableOp/reg_net_hidden_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
reg_net/hidden_4/MatMulMatMul#reg_net/hidden_3/Relu:activations:0.reg_net/hidden_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'reg_net/hidden_4/BiasAdd/ReadVariableOpReadVariableOp0reg_net_hidden_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
reg_net/hidden_4/BiasAddBiasAdd!reg_net/hidden_4/MatMul:product:0/reg_net/hidden_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
reg_net/hidden_4/ReluRelu!reg_net/hidden_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*reg_net/output_layer/MatMul/ReadVariableOpReadVariableOp3reg_net_output_layer_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
reg_net/output_layer/MatMulMatMul#reg_net/hidden_4/Relu:activations:02reg_net/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+reg_net/output_layer/BiasAdd/ReadVariableOpReadVariableOp4reg_net_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
reg_net/output_layer/BiasAddBiasAdd%reg_net/output_layer/MatMul:product:03reg_net/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%reg_net/output_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^reg_net/batch_norm/batchnorm/ReadVariableOp.^reg_net/batch_norm/batchnorm/ReadVariableOp_1.^reg_net/batch_norm/batchnorm/ReadVariableOp_20^reg_net/batch_norm/batchnorm/mul/ReadVariableOp(^reg_net/hidden_1/BiasAdd/ReadVariableOp'^reg_net/hidden_1/MatMul/ReadVariableOp(^reg_net/hidden_2/BiasAdd/ReadVariableOp'^reg_net/hidden_2/MatMul/ReadVariableOp(^reg_net/hidden_3/BiasAdd/ReadVariableOp'^reg_net/hidden_3/MatMul/ReadVariableOp(^reg_net/hidden_4/BiasAdd/ReadVariableOp'^reg_net/hidden_4/MatMul/ReadVariableOp,^reg_net/output_layer/BiasAdd/ReadVariableOp+^reg_net/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2Z
+reg_net/batch_norm/batchnorm/ReadVariableOp+reg_net/batch_norm/batchnorm/ReadVariableOp2^
-reg_net/batch_norm/batchnorm/ReadVariableOp_1-reg_net/batch_norm/batchnorm/ReadVariableOp_12^
-reg_net/batch_norm/batchnorm/ReadVariableOp_2-reg_net/batch_norm/batchnorm/ReadVariableOp_22b
/reg_net/batch_norm/batchnorm/mul/ReadVariableOp/reg_net/batch_norm/batchnorm/mul/ReadVariableOp2R
'reg_net/hidden_1/BiasAdd/ReadVariableOp'reg_net/hidden_1/BiasAdd/ReadVariableOp2P
&reg_net/hidden_1/MatMul/ReadVariableOp&reg_net/hidden_1/MatMul/ReadVariableOp2R
'reg_net/hidden_2/BiasAdd/ReadVariableOp'reg_net/hidden_2/BiasAdd/ReadVariableOp2P
&reg_net/hidden_2/MatMul/ReadVariableOp&reg_net/hidden_2/MatMul/ReadVariableOp2R
'reg_net/hidden_3/BiasAdd/ReadVariableOp'reg_net/hidden_3/BiasAdd/ReadVariableOp2P
&reg_net/hidden_3/MatMul/ReadVariableOp&reg_net/hidden_3/MatMul/ReadVariableOp2R
'reg_net/hidden_4/BiasAdd/ReadVariableOp'reg_net/hidden_4/BiasAdd/ReadVariableOp2P
&reg_net/hidden_4/MatMul/ReadVariableOp&reg_net/hidden_4/MatMul/ReadVariableOp2Z
+reg_net/output_layer/BiasAdd/ReadVariableOp+reg_net/output_layer/BiasAdd/ReadVariableOp2X
*reg_net/output_layer/MatMul/ReadVariableOp*reg_net/output_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
D__inference_hidden_1_layer_call_and_return_conditional_losses_343109

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1hidden_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"hidden_1/kernel/Regularizer/SquareSquare9hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�r
!hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_1/kernel/Regularizer/SumSum&hidden_1/kernel/Regularizer/Square:y:0*hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_1/kernel/Regularizer/mulMul*hidden_1/kernel/Regularizer/mul/x:output:0(hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^hidden_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1hidden_1/kernel/Regularizer/Square/ReadVariableOp1hidden_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_batch_norm_layer_call_fn_343905

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_343018o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�B
�
C__inference_reg_net_layer_call_and_return_conditional_losses_343591
input_layer
batch_norm_343532:
batch_norm_343534:
batch_norm_343536:
batch_norm_343538:"
hidden_1_343541:	�
hidden_1_343543:	�#
hidden_2_343546:
��
hidden_2_343548:	�#
hidden_3_343551:
��
hidden_3_343553:	�#
hidden_4_343556:
��
hidden_4_343558:	�&
output_layer_343561:	�!
output_layer_343563:
identity��"batch_norm/StatefulPartitionedCall� hidden_1/StatefulPartitionedCall�1hidden_1/kernel/Regularizer/Square/ReadVariableOp� hidden_2/StatefulPartitionedCall�1hidden_2/kernel/Regularizer/Square/ReadVariableOp� hidden_3/StatefulPartitionedCall�1hidden_3/kernel/Regularizer/Square/ReadVariableOp� hidden_4/StatefulPartitionedCall�1hidden_4/kernel/Regularizer/Square/ReadVariableOp�$output_layer/StatefulPartitionedCall�
"batch_norm/StatefulPartitionedCallStatefulPartitionedCallinput_layerbatch_norm_343532batch_norm_343534batch_norm_343536batch_norm_343538*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_343065�
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0hidden_1_343541hidden_1_343543*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_343109�
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_343546hidden_2_343548*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_343132�
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_343551hidden_3_343553*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_3_layer_call_and_return_conditional_losses_343155�
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_343556hidden_4_343558*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_4_layer_call_and_return_conditional_losses_343178�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0output_layer_343561output_layer_343563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_343194�
1hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_1_343541*
_output_shapes
:	�*
dtype0�
"hidden_1/kernel/Regularizer/SquareSquare9hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�r
!hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_1/kernel/Regularizer/SumSum&hidden_1/kernel/Regularizer/Square:y:0*hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_1/kernel/Regularizer/mulMul*hidden_1/kernel/Regularizer/mul/x:output:0(hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_2_343546* 
_output_shapes
:
��*
dtype0�
"hidden_2/kernel/Regularizer/SquareSquare9hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_2/kernel/Regularizer/SumSum&hidden_2/kernel/Regularizer/Square:y:0*hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_2/kernel/Regularizer/mulMul*hidden_2/kernel/Regularizer/mul/x:output:0(hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_3_343551* 
_output_shapes
:
��*
dtype0�
"hidden_3/kernel/Regularizer/SquareSquare9hidden_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_3/kernel/Regularizer/SumSum&hidden_3/kernel/Regularizer/Square:y:0*hidden_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_3/kernel/Regularizer/mulMul*hidden_3/kernel/Regularizer/mul/x:output:0(hidden_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_4_343556* 
_output_shapes
:
��*
dtype0�
"hidden_4/kernel/Regularizer/SquareSquare9hidden_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_4/kernel/Regularizer/SumSum&hidden_4/kernel/Regularizer/Square:y:0*hidden_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_4/kernel/Regularizer/mulMul*hidden_4/kernel/Regularizer/mul/x:output:0(hidden_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^batch_norm/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall2^hidden_1/kernel/Regularizer/Square/ReadVariableOp!^hidden_2/StatefulPartitionedCall2^hidden_2/kernel/Regularizer/Square/ReadVariableOp!^hidden_3/StatefulPartitionedCall2^hidden_3/kernel/Regularizer/Square/ReadVariableOp!^hidden_4/StatefulPartitionedCall2^hidden_4/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2f
1hidden_1/kernel/Regularizer/Square/ReadVariableOp1hidden_1/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2f
1hidden_2/kernel/Regularizer/Square/ReadVariableOp1hidden_2/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2f
1hidden_3/kernel/Regularizer/Square/ReadVariableOp1hidden_3/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2f
1hidden_4/kernel/Regularizer/Square/ReadVariableOp1hidden_4/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
__inference_loss_fn_1_344141N
:hidden_2_kernel_regularizer_square_readvariableop_resource:
��
identity��1hidden_2/kernel/Regularizer/Square/ReadVariableOp�
1hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:hidden_2_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_2/kernel/Regularizer/SquareSquare9hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_2/kernel/Regularizer/SumSum&hidden_2/kernel/Regularizer/Square:y:0*hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_2/kernel/Regularizer/mulMul*hidden_2/kernel/Regularizer/mul/x:output:0(hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#hidden_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^hidden_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1hidden_2/kernel/Regularizer/Square/ReadVariableOp1hidden_2/kernel/Regularizer/Square/ReadVariableOp
�
�
-__inference_output_layer_layer_call_fn_344109

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_343194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_batch_norm_layer_call_fn_343918

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_343065o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_hidden_3_layer_call_fn_344051

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_3_layer_call_and_return_conditional_losses_343155p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
F__inference_batch_norm_layer_call_and_return_conditional_losses_343065

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_hidden_2_layer_call_and_return_conditional_losses_343132

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1hidden_2/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_2/kernel/Regularizer/SquareSquare9hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_2/kernel/Regularizer/SumSum&hidden_2/kernel/Regularizer/Square:y:0*hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_2/kernel/Regularizer/mulMul*hidden_2/kernel/Regularizer/mul/x:output:0(hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^hidden_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1hidden_2/kernel/Regularizer/Square/ReadVariableOp1hidden_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
H__inference_output_layer_layer_call_and_return_conditional_losses_344119

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�e
�
__inference__traced_save_344342
file_prefix/
+savev2_batch_norm_gamma_read_readvariableop.
*savev2_batch_norm_beta_read_readvariableop5
1savev2_batch_norm_moving_mean_read_readvariableop9
5savev2_batch_norm_moving_variance_read_readvariableop.
*savev2_hidden_1_kernel_read_readvariableop,
(savev2_hidden_1_bias_read_readvariableop.
*savev2_hidden_2_kernel_read_readvariableop,
(savev2_hidden_2_bias_read_readvariableop.
*savev2_hidden_3_kernel_read_readvariableop,
(savev2_hidden_3_bias_read_readvariableop.
*savev2_hidden_4_kernel_read_readvariableop,
(savev2_hidden_4_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop*
&savev2_squared_sum_read_readvariableop"
savev2_sum_read_readvariableop'
#savev2_residual_read_readvariableop&
"savev2_count_2_read_readvariableop*
&savev2_num_samples_read_readvariableop6
2savev2_adam_batch_norm_gamma_m_read_readvariableop5
1savev2_adam_batch_norm_beta_m_read_readvariableop5
1savev2_adam_hidden_1_kernel_m_read_readvariableop3
/savev2_adam_hidden_1_bias_m_read_readvariableop5
1savev2_adam_hidden_2_kernel_m_read_readvariableop3
/savev2_adam_hidden_2_bias_m_read_readvariableop5
1savev2_adam_hidden_3_kernel_m_read_readvariableop3
/savev2_adam_hidden_3_bias_m_read_readvariableop5
1savev2_adam_hidden_4_kernel_m_read_readvariableop3
/savev2_adam_hidden_4_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop6
2savev2_adam_batch_norm_gamma_v_read_readvariableop5
1savev2_adam_batch_norm_beta_v_read_readvariableop5
1savev2_adam_hidden_1_kernel_v_read_readvariableop3
/savev2_adam_hidden_1_bias_v_read_readvariableop5
1savev2_adam_hidden_2_kernel_v_read_readvariableop3
/savev2_adam_hidden_2_bias_v_read_readvariableop5
1savev2_adam_hidden_3_kernel_v_read_readvariableop3
/savev2_adam_hidden_3_bias_v_read_readvariableop5
1savev2_adam_hidden_4_kernel_v_read_readvariableop3
/savev2_adam_hidden_4_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/squared_sum/.ATTRIBUTES/VARIABLE_VALUEB2keras_api/metrics/2/sum/.ATTRIBUTES/VARIABLE_VALUEB7keras_api/metrics/2/residual/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/num_samples/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_batch_norm_gamma_read_readvariableop*savev2_batch_norm_beta_read_readvariableop1savev2_batch_norm_moving_mean_read_readvariableop5savev2_batch_norm_moving_variance_read_readvariableop*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop*savev2_hidden_2_kernel_read_readvariableop(savev2_hidden_2_bias_read_readvariableop*savev2_hidden_3_kernel_read_readvariableop(savev2_hidden_3_bias_read_readvariableop*savev2_hidden_4_kernel_read_readvariableop(savev2_hidden_4_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop&savev2_squared_sum_read_readvariableopsavev2_sum_read_readvariableop#savev2_residual_read_readvariableop"savev2_count_2_read_readvariableop&savev2_num_samples_read_readvariableop2savev2_adam_batch_norm_gamma_m_read_readvariableop1savev2_adam_batch_norm_beta_m_read_readvariableop1savev2_adam_hidden_1_kernel_m_read_readvariableop/savev2_adam_hidden_1_bias_m_read_readvariableop1savev2_adam_hidden_2_kernel_m_read_readvariableop/savev2_adam_hidden_2_bias_m_read_readvariableop1savev2_adam_hidden_3_kernel_m_read_readvariableop/savev2_adam_hidden_3_bias_m_read_readvariableop1savev2_adam_hidden_4_kernel_m_read_readvariableop/savev2_adam_hidden_4_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop2savev2_adam_batch_norm_gamma_v_read_readvariableop1savev2_adam_batch_norm_beta_v_read_readvariableop1savev2_adam_hidden_1_kernel_v_read_readvariableop/savev2_adam_hidden_1_bias_v_read_readvariableop1savev2_adam_hidden_2_kernel_v_read_readvariableop/savev2_adam_hidden_2_bias_v_read_readvariableop1savev2_adam_hidden_3_kernel_v_read_readvariableop/savev2_adam_hidden_3_bias_v_read_readvariableop1savev2_adam_hidden_4_kernel_v_read_readvariableop/savev2_adam_hidden_4_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *C
dtypes9
725	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::::	�:�:
��:�:
��:�:
��:�:	�:: : : : : : : : : ::::: :::	�:�:
��:�:
��:�:
��:�:	�::::	�:�:
��:�:
��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&	"
 
_output_shapes
:
��:!


_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�:! 

_output_shapes	
:�:&!"
 
_output_shapes
:
��:!"

_output_shapes	
:�:&#"
 
_output_shapes
:
��:!$

_output_shapes	
:�:&%"
 
_output_shapes
:
��:!&

_output_shapes	
:�:%'!

_output_shapes
:	�: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
::%+!

_output_shapes
:	�:!,

_output_shapes	
:�:&-"
 
_output_shapes
:
��:!.

_output_shapes	
:�:&/"
 
_output_shapes
:
��:!0

_output_shapes	
:�:&1"
 
_output_shapes
:
��:!2

_output_shapes	
:�:%3!

_output_shapes
:	�: 4

_output_shapes
::5

_output_shapes
: 
�
�
)__inference_hidden_2_layer_call_fn_344019

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_343132p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�B
�
C__inference_reg_net_layer_call_and_return_conditional_losses_343225

inputs
batch_norm_343083:
batch_norm_343085:
batch_norm_343087:
batch_norm_343089:"
hidden_1_343110:	�
hidden_1_343112:	�#
hidden_2_343133:
��
hidden_2_343135:	�#
hidden_3_343156:
��
hidden_3_343158:	�#
hidden_4_343179:
��
hidden_4_343181:	�&
output_layer_343195:	�!
output_layer_343197:
identity��"batch_norm/StatefulPartitionedCall� hidden_1/StatefulPartitionedCall�1hidden_1/kernel/Regularizer/Square/ReadVariableOp� hidden_2/StatefulPartitionedCall�1hidden_2/kernel/Regularizer/Square/ReadVariableOp� hidden_3/StatefulPartitionedCall�1hidden_3/kernel/Regularizer/Square/ReadVariableOp� hidden_4/StatefulPartitionedCall�1hidden_4/kernel/Regularizer/Square/ReadVariableOp�$output_layer/StatefulPartitionedCall�
"batch_norm/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_norm_343083batch_norm_343085batch_norm_343087batch_norm_343089*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_343018�
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0hidden_1_343110hidden_1_343112*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_343109�
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_343133hidden_2_343135*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_343132�
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_343156hidden_3_343158*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_3_layer_call_and_return_conditional_losses_343155�
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_343179hidden_4_343181*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_4_layer_call_and_return_conditional_losses_343178�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0output_layer_343195output_layer_343197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_343194�
1hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_1_343110*
_output_shapes
:	�*
dtype0�
"hidden_1/kernel/Regularizer/SquareSquare9hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�r
!hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_1/kernel/Regularizer/SumSum&hidden_1/kernel/Regularizer/Square:y:0*hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_1/kernel/Regularizer/mulMul*hidden_1/kernel/Regularizer/mul/x:output:0(hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_2_343133* 
_output_shapes
:
��*
dtype0�
"hidden_2/kernel/Regularizer/SquareSquare9hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_2/kernel/Regularizer/SumSum&hidden_2/kernel/Regularizer/Square:y:0*hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_2/kernel/Regularizer/mulMul*hidden_2/kernel/Regularizer/mul/x:output:0(hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_3_343156* 
_output_shapes
:
��*
dtype0�
"hidden_3/kernel/Regularizer/SquareSquare9hidden_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_3/kernel/Regularizer/SumSum&hidden_3/kernel/Regularizer/Square:y:0*hidden_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_3/kernel/Regularizer/mulMul*hidden_3/kernel/Regularizer/mul/x:output:0(hidden_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_4_343179* 
_output_shapes
:
��*
dtype0�
"hidden_4/kernel/Regularizer/SquareSquare9hidden_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_4/kernel/Regularizer/SumSum&hidden_4/kernel/Regularizer/Square:y:0*hidden_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_4/kernel/Regularizer/mulMul*hidden_4/kernel/Regularizer/mul/x:output:0(hidden_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^batch_norm/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall2^hidden_1/kernel/Regularizer/Square/ReadVariableOp!^hidden_2/StatefulPartitionedCall2^hidden_2/kernel/Regularizer/Square/ReadVariableOp!^hidden_3/StatefulPartitionedCall2^hidden_3/kernel/Regularizer/Square/ReadVariableOp!^hidden_4/StatefulPartitionedCall2^hidden_4/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2f
1hidden_1/kernel/Regularizer/Square/ReadVariableOp1hidden_1/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2f
1hidden_2/kernel/Regularizer/Square/ReadVariableOp1hidden_2/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2f
1hidden_3/kernel/Regularizer/Square/ReadVariableOp1hidden_3/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2f
1hidden_4/kernel/Regularizer/Square/ReadVariableOp1hidden_4/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_hidden_2_layer_call_and_return_conditional_losses_344036

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1hidden_2/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_2/kernel/Regularizer/SquareSquare9hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_2/kernel/Regularizer/SumSum&hidden_2/kernel/Regularizer/Square:y:0*hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_2/kernel/Regularizer/mulMul*hidden_2/kernel/Regularizer/mul/x:output:0(hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^hidden_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1hidden_2/kernel/Regularizer/Square/ReadVariableOp1hidden_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_reg_net_layer_call_fn_343467
input_layer
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reg_net_layer_call_and_return_conditional_losses_343403o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
D__inference_hidden_1_layer_call_and_return_conditional_losses_344004

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1hidden_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"hidden_1/kernel/Regularizer/SquareSquare9hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�r
!hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_1/kernel/Regularizer/SumSum&hidden_1/kernel/Regularizer/Square:y:0*hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_1/kernel/Regularizer/mulMul*hidden_1/kernel/Regularizer/mul/x:output:0(hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^hidden_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1hidden_1/kernel/Regularizer/Square/ReadVariableOp1hidden_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_344163N
:hidden_4_kernel_regularizer_square_readvariableop_resource:
��
identity��1hidden_4/kernel/Regularizer/Square/ReadVariableOp�
1hidden_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:hidden_4_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_4/kernel/Regularizer/SquareSquare9hidden_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_4/kernel/Regularizer/SumSum&hidden_4/kernel/Regularizer/Square:y:0*hidden_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_4/kernel/Regularizer/mulMul*hidden_4/kernel/Regularizer/mul/x:output:0(hidden_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#hidden_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^hidden_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1hidden_4/kernel/Regularizer/Square/ReadVariableOp1hidden_4/kernel/Regularizer/Square/ReadVariableOp
�
�
D__inference_hidden_3_layer_call_and_return_conditional_losses_343155

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1hidden_3/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1hidden_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_3/kernel/Regularizer/SquareSquare9hidden_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_3/kernel/Regularizer/SumSum&hidden_3/kernel/Regularizer/Square:y:0*hidden_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_3/kernel/Regularizer/mulMul*hidden_3/kernel/Regularizer/mul/x:output:0(hidden_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^hidden_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1hidden_3/kernel/Regularizer/Square/ReadVariableOp1hidden_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�B
�
C__inference_reg_net_layer_call_and_return_conditional_losses_343403

inputs
batch_norm_343344:
batch_norm_343346:
batch_norm_343348:
batch_norm_343350:"
hidden_1_343353:	�
hidden_1_343355:	�#
hidden_2_343358:
��
hidden_2_343360:	�#
hidden_3_343363:
��
hidden_3_343365:	�#
hidden_4_343368:
��
hidden_4_343370:	�&
output_layer_343373:	�!
output_layer_343375:
identity��"batch_norm/StatefulPartitionedCall� hidden_1/StatefulPartitionedCall�1hidden_1/kernel/Regularizer/Square/ReadVariableOp� hidden_2/StatefulPartitionedCall�1hidden_2/kernel/Regularizer/Square/ReadVariableOp� hidden_3/StatefulPartitionedCall�1hidden_3/kernel/Regularizer/Square/ReadVariableOp� hidden_4/StatefulPartitionedCall�1hidden_4/kernel/Regularizer/Square/ReadVariableOp�$output_layer/StatefulPartitionedCall�
"batch_norm/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_norm_343344batch_norm_343346batch_norm_343348batch_norm_343350*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_343065�
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0hidden_1_343353hidden_1_343355*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_343109�
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_343358hidden_2_343360*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_343132�
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_343363hidden_3_343365*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_3_layer_call_and_return_conditional_losses_343155�
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_343368hidden_4_343370*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_4_layer_call_and_return_conditional_losses_343178�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0output_layer_343373output_layer_343375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_343194�
1hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_1_343353*
_output_shapes
:	�*
dtype0�
"hidden_1/kernel/Regularizer/SquareSquare9hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�r
!hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_1/kernel/Regularizer/SumSum&hidden_1/kernel/Regularizer/Square:y:0*hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_1/kernel/Regularizer/mulMul*hidden_1/kernel/Regularizer/mul/x:output:0(hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_2_343358* 
_output_shapes
:
��*
dtype0�
"hidden_2/kernel/Regularizer/SquareSquare9hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_2/kernel/Regularizer/SumSum&hidden_2/kernel/Regularizer/Square:y:0*hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_2/kernel/Regularizer/mulMul*hidden_2/kernel/Regularizer/mul/x:output:0(hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_3_343363* 
_output_shapes
:
��*
dtype0�
"hidden_3/kernel/Regularizer/SquareSquare9hidden_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_3/kernel/Regularizer/SumSum&hidden_3/kernel/Regularizer/Square:y:0*hidden_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_3/kernel/Regularizer/mulMul*hidden_3/kernel/Regularizer/mul/x:output:0(hidden_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_4_343368* 
_output_shapes
:
��*
dtype0�
"hidden_4/kernel/Regularizer/SquareSquare9hidden_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_4/kernel/Regularizer/SumSum&hidden_4/kernel/Regularizer/Square:y:0*hidden_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_4/kernel/Regularizer/mulMul*hidden_4/kernel/Regularizer/mul/x:output:0(hidden_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^batch_norm/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall2^hidden_1/kernel/Regularizer/Square/ReadVariableOp!^hidden_2/StatefulPartitionedCall2^hidden_2/kernel/Regularizer/Square/ReadVariableOp!^hidden_3/StatefulPartitionedCall2^hidden_3/kernel/Regularizer/Square/ReadVariableOp!^hidden_4/StatefulPartitionedCall2^hidden_4/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2f
1hidden_1/kernel/Regularizer/Square/ReadVariableOp1hidden_1/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2f
1hidden_2/kernel/Regularizer/Square/ReadVariableOp1hidden_2/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2f
1hidden_3/kernel/Regularizer/Square/ReadVariableOp1hidden_3/kernel/Regularizer/Square/ReadVariableOp2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2f
1hidden_4/kernel/Regularizer/Square/ReadVariableOp1hidden_4/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_batch_norm_layer_call_and_return_conditional_losses_343938

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_reg_net_layer_call_fn_343722

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reg_net_layer_call_and_return_conditional_losses_343403o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_reg_net_layer_call_fn_343256
input_layer
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reg_net_layer_call_and_return_conditional_losses_343225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
D__inference_hidden_4_layer_call_and_return_conditional_losses_343178

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1hidden_4/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1hidden_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_4/kernel/Regularizer/SquareSquare9hidden_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_4/kernel/Regularizer/SumSum&hidden_4/kernel/Regularizer/Square:y:0*hidden_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_4/kernel/Regularizer/mulMul*hidden_4/kernel/Regularizer/mul/x:output:0(hidden_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^hidden_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1hidden_4/kernel/Regularizer/Square/ReadVariableOp1hidden_4/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_hidden_3_layer_call_and_return_conditional_losses_344068

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1hidden_3/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1hidden_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_3/kernel/Regularizer/SquareSquare9hidden_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_3/kernel/Regularizer/SumSum&hidden_3/kernel/Regularizer/Square:y:0*hidden_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_3/kernel/Regularizer/mulMul*hidden_3/kernel/Regularizer/mul/x:output:0(hidden_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^hidden_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1hidden_3/kernel/Regularizer/Square/ReadVariableOp1hidden_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
H__inference_output_layer_layer_call_and_return_conditional_losses_343194

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_hidden_4_layer_call_fn_344083

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_hidden_4_layer_call_and_return_conditional_losses_343178p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_hidden_4_layer_call_and_return_conditional_losses_344100

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1hidden_4/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1hidden_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_4/kernel/Regularizer/SquareSquare9hidden_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_4/kernel/Regularizer/SumSum&hidden_4/kernel/Regularizer/Square:y:0*hidden_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_4/kernel/Regularizer/mulMul*hidden_4/kernel/Regularizer/mul/x:output:0(hidden_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^hidden_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1hidden_4/kernel/Regularizer/Square/ReadVariableOp1hidden_4/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�a
�
C__inference_reg_net_layer_call_and_return_conditional_losses_343800

inputs:
,batch_norm_batchnorm_readvariableop_resource:>
0batch_norm_batchnorm_mul_readvariableop_resource:<
.batch_norm_batchnorm_readvariableop_1_resource:<
.batch_norm_batchnorm_readvariableop_2_resource::
'hidden_1_matmul_readvariableop_resource:	�7
(hidden_1_biasadd_readvariableop_resource:	�;
'hidden_2_matmul_readvariableop_resource:
��7
(hidden_2_biasadd_readvariableop_resource:	�;
'hidden_3_matmul_readvariableop_resource:
��7
(hidden_3_biasadd_readvariableop_resource:	�;
'hidden_4_matmul_readvariableop_resource:
��7
(hidden_4_biasadd_readvariableop_resource:	�>
+output_layer_matmul_readvariableop_resource:	�:
,output_layer_biasadd_readvariableop_resource:
identity��#batch_norm/batchnorm/ReadVariableOp�%batch_norm/batchnorm/ReadVariableOp_1�%batch_norm/batchnorm/ReadVariableOp_2�'batch_norm/batchnorm/mul/ReadVariableOp�hidden_1/BiasAdd/ReadVariableOp�hidden_1/MatMul/ReadVariableOp�1hidden_1/kernel/Regularizer/Square/ReadVariableOp�hidden_2/BiasAdd/ReadVariableOp�hidden_2/MatMul/ReadVariableOp�1hidden_2/kernel/Regularizer/Square/ReadVariableOp�hidden_3/BiasAdd/ReadVariableOp�hidden_3/MatMul/ReadVariableOp�1hidden_3/kernel/Regularizer/Square/ReadVariableOp�hidden_4/BiasAdd/ReadVariableOp�hidden_4/MatMul/ReadVariableOp�1hidden_4/kernel/Regularizer/Square/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
#batch_norm/batchnorm/ReadVariableOpReadVariableOp,batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0_
batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_norm/batchnorm/addAddV2+batch_norm/batchnorm/ReadVariableOp:value:0#batch_norm/batchnorm/add/y:output:0*
T0*
_output_shapes
:f
batch_norm/batchnorm/RsqrtRsqrtbatch_norm/batchnorm/add:z:0*
T0*
_output_shapes
:�
'batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp0batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batch_norm/batchnorm/mulMulbatch_norm/batchnorm/Rsqrt:y:0/batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:y
batch_norm/batchnorm/mul_1Mulinputsbatch_norm/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_norm/batchnorm/ReadVariableOp_1ReadVariableOp.batch_norm_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
batch_norm/batchnorm/mul_2Mul-batch_norm/batchnorm/ReadVariableOp_1:value:0batch_norm/batchnorm/mul:z:0*
T0*
_output_shapes
:�
%batch_norm/batchnorm/ReadVariableOp_2ReadVariableOp.batch_norm_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
batch_norm/batchnorm/subSub-batch_norm/batchnorm/ReadVariableOp_2:value:0batch_norm/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
batch_norm/batchnorm/add_1AddV2batch_norm/batchnorm/mul_1:z:0batch_norm/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
hidden_1/MatMulMatMulbatch_norm/batchnorm/add_1:z:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hidden_2/MatMulMatMulhidden_1/Relu:activations:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hidden_3/MatMulMatMulhidden_2/Relu:activations:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
hidden_3/ReluReluhidden_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
hidden_4/MatMul/ReadVariableOpReadVariableOp'hidden_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hidden_4/MatMulMatMulhidden_3/Relu:activations:0&hidden_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
hidden_4/BiasAdd/ReadVariableOpReadVariableOp(hidden_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
hidden_4/BiasAddBiasAddhidden_4/MatMul:product:0'hidden_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
hidden_4/ReluReluhidden_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
output_layer/MatMulMatMulhidden_4/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1hidden_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"hidden_1/kernel/Regularizer/SquareSquare9hidden_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�r
!hidden_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_1/kernel/Regularizer/SumSum&hidden_1/kernel/Regularizer/Square:y:0*hidden_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_1/kernel/Regularizer/mulMul*hidden_1/kernel/Regularizer/mul/x:output:0(hidden_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_2/kernel/Regularizer/SquareSquare9hidden_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_2/kernel/Regularizer/SumSum&hidden_2/kernel/Regularizer/Square:y:0*hidden_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_2/kernel/Regularizer/mulMul*hidden_2/kernel/Regularizer/mul/x:output:0(hidden_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_3/kernel/Regularizer/SquareSquare9hidden_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_3/kernel/Regularizer/SumSum&hidden_3/kernel/Regularizer/Square:y:0*hidden_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_3/kernel/Regularizer/mulMul*hidden_3/kernel/Regularizer/mul/x:output:0(hidden_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1hidden_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'hidden_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"hidden_4/kernel/Regularizer/SquareSquare9hidden_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!hidden_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
hidden_4/kernel/Regularizer/SumSum&hidden_4/kernel/Regularizer/Square:y:0*hidden_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!hidden_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;�
hidden_4/kernel/Regularizer/mulMul*hidden_4/kernel/Regularizer/mul/x:output:0(hidden_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: l
IdentityIdentityoutput_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^batch_norm/batchnorm/ReadVariableOp&^batch_norm/batchnorm/ReadVariableOp_1&^batch_norm/batchnorm/ReadVariableOp_2(^batch_norm/batchnorm/mul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp2^hidden_1/kernel/Regularizer/Square/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp2^hidden_2/kernel/Regularizer/Square/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp2^hidden_3/kernel/Regularizer/Square/ReadVariableOp ^hidden_4/BiasAdd/ReadVariableOp^hidden_4/MatMul/ReadVariableOp2^hidden_4/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2J
#batch_norm/batchnorm/ReadVariableOp#batch_norm/batchnorm/ReadVariableOp2N
%batch_norm/batchnorm/ReadVariableOp_1%batch_norm/batchnorm/ReadVariableOp_12N
%batch_norm/batchnorm/ReadVariableOp_2%batch_norm/batchnorm/ReadVariableOp_22R
'batch_norm/batchnorm/mul/ReadVariableOp'batch_norm/batchnorm/mul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2f
1hidden_1/kernel/Regularizer/Square/ReadVariableOp1hidden_1/kernel/Regularizer/Square/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2f
1hidden_2/kernel/Regularizer/Square/ReadVariableOp1hidden_2/kernel/Regularizer/Square/ReadVariableOp2B
hidden_3/BiasAdd/ReadVariableOphidden_3/BiasAdd/ReadVariableOp2@
hidden_3/MatMul/ReadVariableOphidden_3/MatMul/ReadVariableOp2f
1hidden_3/kernel/Regularizer/Square/ReadVariableOp1hidden_3/kernel/Regularizer/Square/ReadVariableOp2B
hidden_4/BiasAdd/ReadVariableOphidden_4/BiasAdd/ReadVariableOp2@
hidden_4/MatMul/ReadVariableOphidden_4/MatMul/ReadVariableOp2f
1hidden_4/kernel/Regularizer/Square/ReadVariableOp1hidden_4/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_layer4
serving_default_input_layer:0���������@
output_layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_sequential
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

4beta_1

5beta_2
	6decay
7learning_rate
8itermnmompmqmrms"mt#mu(mv)mw.mx/myvzv{v|v}v~v"v�#v�(v�)v�.v�/v�"
	optimizer
�
0
1
2
3
4
5
6
7
"8
#9
(10
)11
.12
/13"
trackable_list_wrapper
v
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
	trainable_variables

regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
:2batch_norm/gamma
:2batch_norm/beta
&:$ (2batch_norm/moving_mean
*:( (2batch_norm/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�2hidden_1/kernel
:�2hidden_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2hidden_2/kernel
:�2hidden_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
 regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2hidden_3/kernel
:�2hidden_3/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2hidden_4/kernel
:�2hidden_4/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
*	variables
+trainable_variables
,regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$	�2output_layer/kernel
:2output_layer/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
0	variables
1trainable_variables
2regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
.
0
1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
5
\0
]1
^2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	_total
	`count
a	variables
b	keras_api"
_tf_keras_metric
N
	ctotal
	dcount
e	variables
f	keras_api"
_tf_keras_metric
�
gsquared_sum
hsum
iresidual
ires
	jcount
knum_samples
l	variables
m	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
_0
`1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
:  (2total
:  (2count
.
c0
d1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
: (2squared_sum
: (2sum
: (2residual
: (2count
:  (2num_samples
C
g0
h1
i2
j3
k4"
trackable_list_wrapper
-
l	variables"
_generic_user_object
#:!2Adam/batch_norm/gamma/m
": 2Adam/batch_norm/beta/m
':%	�2Adam/hidden_1/kernel/m
!:�2Adam/hidden_1/bias/m
(:&
��2Adam/hidden_2/kernel/m
!:�2Adam/hidden_2/bias/m
(:&
��2Adam/hidden_3/kernel/m
!:�2Adam/hidden_3/bias/m
(:&
��2Adam/hidden_4/kernel/m
!:�2Adam/hidden_4/bias/m
+:)	�2Adam/output_layer/kernel/m
$:"2Adam/output_layer/bias/m
#:!2Adam/batch_norm/gamma/v
": 2Adam/batch_norm/beta/v
':%	�2Adam/hidden_1/kernel/v
!:�2Adam/hidden_1/bias/v
(:&
��2Adam/hidden_2/kernel/v
!:�2Adam/hidden_2/bias/v
(:&
��2Adam/hidden_3/kernel/v
!:�2Adam/hidden_3/bias/v
(:&
��2Adam/hidden_4/kernel/v
!:�2Adam/hidden_4/bias/v
+:)	�2Adam/output_layer/kernel/v
$:"2Adam/output_layer/bias/v
�2�
(__inference_reg_net_layer_call_fn_343256
(__inference_reg_net_layer_call_fn_343689
(__inference_reg_net_layer_call_fn_343722
(__inference_reg_net_layer_call_fn_343467�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_reg_net_layer_call_and_return_conditional_losses_343800
C__inference_reg_net_layer_call_and_return_conditional_losses_343892
C__inference_reg_net_layer_call_and_return_conditional_losses_343529
C__inference_reg_net_layer_call_and_return_conditional_losses_343591�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
!__inference__wrapped_model_342994input_layer"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_batch_norm_layer_call_fn_343905
+__inference_batch_norm_layer_call_fn_343918�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_batch_norm_layer_call_and_return_conditional_losses_343938
F__inference_batch_norm_layer_call_and_return_conditional_losses_343972�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_hidden_1_layer_call_fn_343987�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_hidden_1_layer_call_and_return_conditional_losses_344004�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_hidden_2_layer_call_fn_344019�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_hidden_2_layer_call_and_return_conditional_losses_344036�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_hidden_3_layer_call_fn_344051�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_hidden_3_layer_call_and_return_conditional_losses_344068�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_hidden_4_layer_call_fn_344083�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_hidden_4_layer_call_and_return_conditional_losses_344100�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_output_layer_layer_call_fn_344109�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_output_layer_layer_call_and_return_conditional_losses_344119�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_344130�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_344141�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_344152�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_344163�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
$__inference_signature_wrapper_343656input_layer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_342994�"#()./4�1
*�'
%�"
input_layer���������
� ";�8
6
output_layer&�#
output_layer����������
F__inference_batch_norm_layer_call_and_return_conditional_losses_343938b3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
F__inference_batch_norm_layer_call_and_return_conditional_losses_343972b3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
+__inference_batch_norm_layer_call_fn_343905U3�0
)�&
 �
inputs���������
p 
� "�����������
+__inference_batch_norm_layer_call_fn_343918U3�0
)�&
 �
inputs���������
p
� "�����������
D__inference_hidden_1_layer_call_and_return_conditional_losses_344004]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� }
)__inference_hidden_1_layer_call_fn_343987P/�,
%�"
 �
inputs���������
� "������������
D__inference_hidden_2_layer_call_and_return_conditional_losses_344036^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_hidden_2_layer_call_fn_344019Q0�-
&�#
!�
inputs����������
� "������������
D__inference_hidden_3_layer_call_and_return_conditional_losses_344068^"#0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_hidden_3_layer_call_fn_344051Q"#0�-
&�#
!�
inputs����������
� "������������
D__inference_hidden_4_layer_call_and_return_conditional_losses_344100^()0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_hidden_4_layer_call_fn_344083Q()0�-
&�#
!�
inputs����������
� "�����������;
__inference_loss_fn_0_344130�

� 
� "� ;
__inference_loss_fn_1_344141�

� 
� "� ;
__inference_loss_fn_2_344152"�

� 
� "� ;
__inference_loss_fn_3_344163(�

� 
� "� �
H__inference_output_layer_layer_call_and_return_conditional_losses_344119]./0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
-__inference_output_layer_layer_call_fn_344109P./0�-
&�#
!�
inputs����������
� "�����������
C__inference_reg_net_layer_call_and_return_conditional_losses_343529u"#()./<�9
2�/
%�"
input_layer���������
p 

 
� "%�"
�
0���������
� �
C__inference_reg_net_layer_call_and_return_conditional_losses_343591u"#()./<�9
2�/
%�"
input_layer���������
p

 
� "%�"
�
0���������
� �
C__inference_reg_net_layer_call_and_return_conditional_losses_343800p"#()./7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
C__inference_reg_net_layer_call_and_return_conditional_losses_343892p"#()./7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
(__inference_reg_net_layer_call_fn_343256h"#()./<�9
2�/
%�"
input_layer���������
p 

 
� "�����������
(__inference_reg_net_layer_call_fn_343467h"#()./<�9
2�/
%�"
input_layer���������
p

 
� "�����������
(__inference_reg_net_layer_call_fn_343689c"#()./7�4
-�*
 �
inputs���������
p 

 
� "�����������
(__inference_reg_net_layer_call_fn_343722c"#()./7�4
-�*
 �
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_343656�"#()./C�@
� 
9�6
4
input_layer%�"
input_layer���������";�8
6
output_layer&�#
output_layer���������