# import pyttsx3
# engine = pyttsx3.init()
# engine.say('Sally sells seashells by the seashore.')
# engine.say('The quick brown fox jumped over the lazy dog.')
# engine.runAndWait()

import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
   engine.setProperty('voice', voice.id)
   print("voice:",voice)
   for i in range(11):
   		engine.say(i)
#engine.runAndWait()
