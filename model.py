import torch
import torch.nn.functional as F

class DFPBasicModel(torch.nn.Module):
  def __init__(self, sensory_size, measurement_size, timesteps_size, action_size):
    super(DFPBasicModel, self).__init__()
    self.perception1=torch.nn.Conv2d(sensory_size[0], 32, kernel_size=3, stride=1, padding=1)
    self.perception2=torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.perception3=torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    #Determiner la taille d'input de la couche lineaire
    self.perception4=torch.nn.Linear(sensory_size[1]*sensory_size[2], 512)
    
    self.measurement1=torch.nn.Linear(measurement_size, 128)
    self.measurement2=torch.nn.Linear(128, 128)
    self.measurement3=torch.nn.Linear(128,128)

    self.goal1=torch.nn.Linear(measurement_size*timesteps_size, 128)
    self.goal2=torch.nn.Linear(128, 128)
    self.goal3=torch.nn.Linear(128, 128)

    self.expectation1=torch.nn.Linear(512+128+128, 512)
    self.expectation2=torch.nn.Linear(512, measurement_size*timesteps_size)

    self.action1=torch.nn.Linear(512+128+128, 512)
    self.action2=torch.nn.Linear(512, measurement_size*timesteps_size*action_size)

    self.action_size = action_size
    self.sensory_size = sensory_size
    self.measurement_size = measurement_size
    self.timesteps_size = timesteps_size

  def forward(self, sensory_input, measurement_input, goal_input, lambda_relu = 0.05):
    sensory_stream = self.perception1(sensory_input) #TODO: probably doesnt work because of batch size
    sensory_stream = F.max_pool2d(sensory_stream, kernel_size=(2,2))
    sensory_stream = self.perception2(sensory_stream)
    sensory_stream = F.max_pool2d(sensory_stream, kernel_size=(2,2))
    sensory_stream = self.perception3(sensory_stream)
    sensory_stream = F.max_pool2d(sensory_stream, kernel_size=(2,2))
    sensory_stream = self.perception4(sensory_stream.flatten(start_dim=1)) 
    sensory_stream = F.leaky_relu(sensory_stream, lambda_relu)

    measurement_stream = self.measurement1(measurement_input)
    measurement_stream = F.leaky_relu(measurement_stream, lambda_relu)
    measurement_stream = self.measurement2(measurement_stream)
    measurement_stream = F.leaky_relu(measurement_stream, lambda_relu)
    measurement_stream = self.measurement3(measurement_stream)
    measurement_stream = F.leaky_relu(measurement_stream, lambda_relu)

    goal_stream = self.goal1(goal_input) 
    goal_stream = F.leaky_relu(goal_stream, lambda_relu)
    goal_stream = self.goal2(goal_stream)
    goal_stream = F.leaky_relu(goal_stream, lambda_relu)
    goal_stream = self.goal3(goal_stream)
    goal_stream = F.leaky_relu(goal_stream, lambda_relu)

    input_representation = torch.cat((sensory_stream, measurement_stream, goal_stream), 1) #TODO: verify axis
    
    expectation_stream = self.expectation1(input_representation)
    expectation_stream = F.leaky_relu(expectation_stream, lambda_relu)
    expectation_stream = self.expectation2(expectation_stream)
    expectation_stream = F.leaky_relu(expectation_stream, lambda_relu)
    expectation_stream = expectation_stream.repeat((1,self.action_size))

    action_stream = self.action1(input_representation)
    action_stream = F.leaky_relu(action_stream, lambda_relu)
    action_stream = self.action2(action_stream)
    action_stream = F.leaky_relu(action_stream, lambda_relu)
    action_stream = action_stream - torch.mean(action_stream, dim=1, keepdim=True) #normalisation

    output = action_stream + expectation_stream

    return output.view(-1,self.action_size, self.measurement_size*self.timesteps_size)