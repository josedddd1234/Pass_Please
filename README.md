# Pass_Please
My father is a person that cannot see (almost blind), for helping him, I´m trying to create a device that can detect  whether a traffic light is red o green. Thats the reason for the name pass please. Here I will be posting my thinking thoughts and my progression. 

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/0eafed29-c4fd-4b02-b7f8-697b95a66b2d" />

##  Materials_planned
I have planned to use a ESP32_CAM, a gyroscope and a battery holder. Making it affordable and easy to replicate. 

## First_3D model
I have already designed the first prototype, I will be putting the model in github. in adition i developed a CNN for reading the lights, the code is on the github page. The purpose is to made this like and addition to the white cane. The goal is making it sheep and confortable, improving life quality of this poblation

<img width="486" height="563" alt="image" src="https://github.com/user-attachments/assets/d743a579-0666-4528-8ae1-ae7f954eff3e" />


In the following image, it is seen how the ESP32 CAM is introduced in the model
<img width="531" height="576" alt="image" src="https://github.com/user-attachments/assets/5b87268a-d65a-4cbb-9a2d-488bea2d1890" />

I will make a 3D print of it, and see what changes can i make 

## CNN FUNCTION 
I made a CNN using tensorslow with three outputs (red light, green light and no traffic light), using the relu activation function and 25 epocs. The results are good, but the foto needs to be really close for it to work. The model and the two scripts used are in the folder scripts. Here is an example with a foto taken randomly in google earth 

<img width="31" height="87" alt="WhatsApp Image 2026-08-22 at 8 28 25 PM" src="https://github.com/user-attachments/assets/f48bd38f-df34-47ec-a180-7a16474d7594" />

<img width="551" height="140" alt="image" src="https://github.com/user-attachments/assets/bdafd22a-472d-4311-8db9-26ce214c0374" />

The detection is correct but the traffic light needed to be really zoomed in. So i will try another alternatives
