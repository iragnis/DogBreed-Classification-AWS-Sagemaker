# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.


## Project Set Up and Installation
* Enter AWS through the gateway in the course and open SageMaker Studio. 
* Download the starter files.
* Download/Make the dataset available.(we will be using dog-images dataset for this classification model)
* You can use the python3 kernel for better results
* Download all the dependacies

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
<!-- ![s3_upload](https://user-images.githubusercontent.com/97392797/149764473-c0a3a0f0-aa12-4a40-ac40-524ed709f33f.jpg) -->
<img src="https://user-images.githubusercontent.com/97392797/149764473-c0a3a0f0-aa12-4a40-ac40-524ed709f33f.jpg" width="850" height="350">

## HyperParameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
* A pretrained Resnet50 model was used with the output set to 133(as there are 133 classes of dogs in the dataset)
*  Hyper-parameters tuned
    * Learning rate - default(x) is 0.001 , chosen range is ```[0.0001, 0.1]```
    * eps - defaut is 1e-08 , chosen range is``` [1e-9, 1e-8]```
    * Weight decay - default(x) is 0.01 , chosen range is ```[1e-3, 1e-1]```
    * Batch size -- chosen range ```[ 64, 128]```
    * Best Hyper-parameters after tuning:
    
```{'batch_size': 64, 'eps': '7.384494989553586e-09', 'lr': '0.0027464603299771654', 'weight_decay': '0.09605621750189348'} ```

### Hyperparameter Tuning Sagemaker snapshot

<img src="https://user-images.githubusercontent.com/97392797/149766504-121c3d63-8875-4abb-95c3-8f934e088c93.png" width="800" height="350"/>

### HyperParameter Tuning Job 
<img src="https://user-images.githubusercontent.com/97392797/149766985-0b30538e-1c7a-413d-97a6-cfcff0a6f61a.jpg" width="850" height="300"/>

### Multiple training jobs triggered by the HyperParameter Tuning Job

<img src="https://user-images.githubusercontent.com/97392797/149769452-bf1db0de-a419-414b-835f-693b755ccb71.png" width="850" height="300"/>
### Best HyperParameter training job Summary

![best_training_jobsummary](https://user-images.githubusercontent.com/97392797/149769352-9c66122b-b9ff-41f1-b094-8f5ee710422e.jpg)
###  Best HyperParameter training job Logs
![best_training_job_logs](https://user-images.githubusercontent.com/97392797/149770283-79ac8420-218d-4aef-8310-658b7709935d.jpg)

## Debugging and Profiling
We had set the Debugger hook to record and keep track of the Loss Criterion metrics of the process in training and validation/testing phases. The Plot of the Cross entropy loss is shown below:

<img src="https://user-images.githubusercontent.com/97392797/149770676-e6271fdf-58ca-478f-b37f-f4a13925813a.jpg" height="400" width="400"/>

### Results(Future work)
As seen above the validation graph seems to have a higer loss which can obviously be worked on by either adding more fully-connected layers
or changing the pretrained model used for tranfer learning

## Endpoint Metrics
![endpoint_instance_util](https://user-images.githubusercontent.com/97392797/149771866-664e607f-585a-4411-a7ed-87e550fa04b7.jpg)
### Results 
Results look pretty good, as we had utilized the GPU while hyperparameter tuning and training of the fine-tuned ResNet50 model. We used the ml.g4dn.xlarge instance type for the runing the traiing purposes. However while deploying the model to an endpoint we used the "ml.t2.medium" instance type to save cost and resources.

## Model Deployment
* Model was deployed to a "ml.t2.medium" instance type and we used the "endpoint_inference.py" script to setup and deploy our working endpoint.
* For testing purposes , we will be using some test images that we have stored in the "testImages" folder.
* We will be reading in some test images from the folder and try to send those images as input and invoke our deployed endpoint
## Deployed Active Endpoint Snapshot
![end_point](https://user-images.githubusercontent.com/97392797/149772794-cf34a3d8-dc8c-471e-9c16-6e6ad76142dc.png)


## Deployed Active Endpoint Logs showing success (200) status code received
![1](https://user-images.githubusercontent.com/97392797/149773822-bc5cd39b-dc83-4f0a-a852-efd30dbad09f.jpg)


## Test cell used for testing the Endpoint Snapshot
![deploying_endpoint](https://user-images.githubusercontent.com/97392797/149772688-2d4ff1bf-df49-41de-8fcc-5448eb5b5812.png)
## Sample output returned from endpoint Snapshot
![endpoint_result](https://user-images.githubusercontent.com/97392797/149772847-3d98055a-16eb-4a25-8a7a-693c7453e0a2.png)

## Future Work(Summary)
 *How can you improve the performance of your model?*
 * I believe there's scope for improvement,given more time I would like to try out other pre-trained models along with different number of neurons.
 * I would also like to experiment with different number of fully-connected layers
 * The combined effect of the above two mentioned suggestions is bound to show an improvement in the model's ability to classify the images correctly
 



