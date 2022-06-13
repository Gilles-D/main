    /* 
      IR Breakbeam sensor demo!
    */
     
    #define LED_BEAM_PIN 3
    #define LED_OPTO_PIN 4
     
    #define SENSORPIN 7
    #define BUTTONPIN 8
     
    // variables will change:
    int sensorState = 0, lastState=0;         // variable for reading the sensor status
    int buttonState = 0, last_buttonState=0; // variable for reading the pushbutton status
     
    void setup() {
      // initialize the LED pin as an output:
      pinMode(LED_BEAM_PIN, OUTPUT);     
      pinMode(LED_OPTO_PIN, OUTPUT); 
      // initialize the sensor pin as an input:
      pinMode(SENSORPIN, INPUT);
      pinMode(BUTTONPIN, INPUT); 
           
      digitalWrite(SENSORPIN, HIGH); // turn on the pullup
      digitalWrite(BUTTONPIN, LOW); // turn on the pullup
      
      Serial.begin(9600);
    }
     
    void loop(){
      // read the state of the pushbutton value:
      sensorState = digitalRead(SENSORPIN);
      buttonState = digitalRead(BUTTONPIN);
     
      // check if the sensor beam is broken
      // if it is, the sensorState is LOW:
      if (sensorState == LOW) {
        Serial.println("Broken");     
        // turn LED on:
        digitalWrite(LED_BEAM_PIN, HIGH);
        delay(1000);
        digitalWrite(LED_OPTO_PIN, HIGH);
        delay(1500);
        digitalWrite(LED_OPTO_PIN, LOW);
        digitalWrite(LED_BEAM_PIN, LOW);
      } 
      else if (buttonState == LOW) {
        Serial.println("Arm"); 
        delay(500);
      }
      else {
        // turn LED off:
        digitalWrite(LED_BEAM_PIN, LOW); 
      }
      
      if (sensorState && !lastState) {
        Serial.println("Unbroken");
      } 
//      if (!sensorState && lastState) {
//        Serial.println("Broken");
//      }
      lastState = sensorState;
    }
