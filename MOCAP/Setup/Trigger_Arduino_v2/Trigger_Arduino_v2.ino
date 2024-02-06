
//////////////////////////////////////////////////////////////////////////////////////////
// Definitions
//////////////////////////////////////////////////////////////////////////////////////////
        
    #define SENSORPIN 7 // Input/Sensor of the IR Beam
    #define REMOTEPIN 10 // Input of the remote receiver

    #define TTL_AMP 9
    #define TTL_OPTO 11

    #define LED_MOCAP 8 //Output LED to sync MOCAP

    #define BUTTON_START  2
    #define BUTTON_STOP 3
    #define BUTTON_RESET 4


    // variables will change:
    int sensorState = 0, lastState=0;         // variable for reading the sensor status

    int Record_Status = 0;
    
    //Add variables for opto frequency


    int buttonState_start = 0;         // variable for reading the pushbutton status
    int buttonState_stop = 0;         // variable for reading the pushbutton status
    int buttonState_reset = 0;         // variable for reading the pushbutton status
    
    
//////////////////////////////////////////////////////////////////////////////////////////
// Setup
//////////////////////////////////////////////////////////////////////////////////////////
     
    void setup() {
      // initialize the LED pin as an output:
      pinMode(LED_MOCAP, OUTPUT);
      digitalWrite(LED_MOCAP, LOW);  

      // initialize the TTL pin as an output:

      pinMode(TTL_AMP, OUTPUT);
      digitalWrite(TTL_AMP, LOW);
      
      pinMode(TTL_OPTO, OUTPUT);
      digitalWrite(TTL_OPTO, LOW);

      // initialize the sensor pin as an input:
      pinMode(SENSORPIN, INPUT);         
      digitalWrite(SENSORPIN, HIGH);
      
      Serial.begin(9600);

      pinMode(BUTTON_START, INPUT);
      pinMode(BUTTON_STOP, INPUT);
      pinMode(BUTTON_RESET, INPUT);
    }

//////////////////////////////////////////////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////////////////////////////////////////////

void Triggered(){
  // The beam is triggered : Turn on opto      20Hz - 20ms pulse
  for (int i = 0; i <= 50; i++) {
    digitalWrite(TTL_OPTO, HIGH);
    delay(20);
    digitalWrite(TTL_OPTO, LOW);
    delay(30);
    Serial.println(i);
  }
}
  


//////////////////////////////////////////////////////////////////////////////////////////
//Loop
//////////////////////////////////////////////////////////////////////////////////////////   
    void loop(){
      // read the state of the IR remote receiver value:
      sensorState = digitalRead(SENSORPIN);

        buttonState_start = digitalRead(BUTTON_START);
        buttonState_stop = digitalRead(BUTTON_STOP);
        buttonState_reset = digitalRead(BUTTON_RESET);

     
      // check if the sensor beam is broken
      // if it is, the sensorState is LOW:
      if (sensorState == LOW) {
        Serial.println("Broken");
        Triggered();
        delay(2000);
      } 
      //Check if the Remote sensor detect Remote Button being pushed
      // If it is, send the button value to serial
      
      else if (buttonState_start == HIGH && Record_Status ==0){

          Serial.println("Start");
          delay(500);
          digitalWrite(LED_MOCAP, HIGH);
          digitalWrite(TTL_AMP, HIGH);
          delay(5);
          digitalWrite(TTL_AMP, LOW);
          delay(500);
          digitalWrite(LED_MOCAP, LOW);
          Record_Status = 1;
      }
      else if (buttonState_stop == HIGH && Record_Status ==1){

          Serial.println("Stop");
          Record_Status = 0;
          delay(2000);
      }
      else if (buttonState_reset == HIGH&& Record_Status ==0){

          Serial.println("Reset");
          Record_Status = 0;
          delay(100);
      }

      //Else, if the beam is not broken, say it, and wait
      if (sensorState && !lastState) {
        Serial.println("Unbroken");
      } 
//      if (!sensorState && lastState) {
//        Serial.println("Broken");
//      }
      lastState = sensorState;

    }
