import numpy as np
import matplotlib.pyplot as plt
import activation_function as af

class NN:
    def __init__(self) -> None:

        """
        lists to be update in neural network are:
        weight
        bias 
        derivatives_activation
        z 

        """
        self.weight = []
        self.bias = []
        self.z = [] 
        self.derivatives_activation = []
        self.derivatives = []
        self.outputs = [] 
        self.forward_ = True
        self.layer_num = 0

    def input_layer(self,input_data):

        if self.forward_ == True:
            self.outputs.append(input_data)

        if self.forward_ == False:
            self.outputs[0]= input_data

        return input_data
    
    def Dense(self,neurons,input_data,activation):
        
        input_shape = np.array(input_data).shape

        if self.forward_ == True:
            weight = np.random.randn(input_shape[-1],neurons)
            bias = np.random.randn(neurons)

            self.weight.append(weight)
            self.bias.append(bias)

            z = np.dot(input_data,weight) + bias

            self.z.append(z)

            D_a_funcation = eval(f"af.D_{activation}")
            a_funcation = eval(f"af.{activation}")

            D_a = D_a_funcation(a_funcation(z))

            self.derivatives_activation.append(D_a)
            self.outputs.append(a_funcation(z))

            deri = np.ones((input_shape[0],neurons))
            self.derivatives.append(deri)

            return a_funcation(z)

        if self.forward_ == False:
            
            if np.mean(self.outputs[-1]) == 0:
               
                self.weight[self.layer_num] =  np.random.randn(input_shape[-1],neurons)        
                self.bias[self.layer_num] =  np.random.randn(neurons)                       

            z = np.dot(self.outputs[self.layer_num],self.weight[self.layer_num]) + self.bias[self.layer_num]
            
            D_a_funcation = eval(f"af.D_{activation}")
            a_funcation = eval(f"af.{activation}")

            D_a = D_a_funcation(a_funcation(z))

            self.outputs[self.layer_num+1] = a_funcation(z)
            self.derivatives_activation[self.layer_num] = D_a

            self.layer_num += 1

            return a_funcation(z)
        
    def loss(self,actual_output):
        return np.mean(0.5*(actual_output-self.outputs[-1])**2)
          

    def backward(self,actual_output,L_R):
        
        for i in range(len(self.weight)):
            r_i = len(self.weight)-i-1

            if i == 0:
                self.derivatives[-1] = 2*(actual_output-self.outputs[-1]) * self.derivatives_activation[-1]
               
            if i > 0:
                self.derivatives[r_i] = np.dot(self.derivatives[r_i+1] , np.array(self.weight[r_i+1]).T)* self.derivatives_activation[r_i]
                
            DL_DW = np.dot( np.array(self.outputs[r_i]).T,self.derivatives[r_i] )

            self.weight[r_i] = self.weight[r_i] + L_R * DL_DW

            self.bias[r_i] = self.bias[r_i] +  L_R * np.sum(self.derivatives[r_i]) 

        self.forward_ = False
        self.layer_num = 0
            
    def predction(self,input_data):
        pred_output = self.outputs.copy()
        pred_output[0] = input_data

        for i in range(len(self.weight)):
            pred_output[i+1] = af.relu(np.dot(pred_output[i],self.weight[i]) + self.bias[i])

        return pred_output[-1]
    
    def forward(self):

        for i in range(len(self.weight)):
            self.z[i] = np.dot(self.outputs[i],self.weight[i]) +self.bias[i]

            self.outputs[i+1] = af.relu(self.z[i])
            self.derivatives_activation[i] = af.D_relu(self.z[i])
        



model = NN()

input_data = np.array([[1],[2],[3],[4]])
actual_out = np.array([[3],[4],[5],[6]])

graph_e =[]
graph_l =[]

for e in range(1000):
    
    input_l = model.input_layer(input_data)
    l1 = model.Dense(2,input_l,"relu")
    l2 = model.Dense(3,l1,"relu")
    prediction = model.Dense(1,l2,"relu")
    if e%100 == 0:
        print("epochs:",e," ","loss:",model.loss(actual_out))
        
    graph_l.append(model.loss(actual_out))
    graph_e.append(e)
        

    model.backward(actual_out,0.001)


print(model.predction([[4]]))

plt.plot(graph_e,graph_l)
plt.show()
