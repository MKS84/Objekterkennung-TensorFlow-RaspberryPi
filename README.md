# Objekterkennung-TensorFlow-RaspberryPi
Realisierung einer Objekterkennung auf einem Raspberry Pi

# Objekterkennung auf einem Raspberry Pi mit TensorFlow

![PiCam](https://github.com/MKS84/Objekterkennung-TensorFlow-RaspberryPi/blob/main/Bilder/Pi_cam.jpg)
![TensorFlow](https://github.com/MKS84/Objekterkennung-TensorFlow-RaspberryPi/blob/main/Bilder/tensorflow2objectdetection.png)

---

SBS Herzogenaurach

FSMT2 Mechatronik

Fach Künstliche Intelligenz

13.04.2021

Markus Kraft, Max Kätscher, Thomas Weisel

---

## 1 Inhaltsverzeichnis

1 Inhaltsverzeichnis  
2 Einleitung  
3 Benötigte Komponenten  
4 Vorbereitung des Raspberry Pis  
5 Installation von TensorFlow  
    5.1 Disclaimer  
    5.2 Python installieren  
    5.3 TensorFlow Lite 2 installieren  
        5.3.1 Voraussetzungen  
        5.3.2 Virtuelle Umgebung  
        5.3.3 rpi-vision installieren  
        5.3.4 TensorFlow 2.x installieren  
    5.4 Ausführen der Graphic Labeling Demo  
    5.5 Anpassen der Bildschirmausgabe  
6 Trainieren der KI  
7 Implementierung der Objekterkennung  
8 Bewertung der KI
9 Literatur- und Abbildungsverzeichnis  
    Websites  
    Bilder  

## 2 Einleitung

Im Rahmen einer Projektarbeit im Fach Künstliche Intelligenz war unser Ziel eine Objekterkennung zu realisieren, welche auf einem Raspberry Pi ablaufen soll. Somit wird am Raspberry Pi im Folgenden eine Kamera angeschlossen, Software installiert und letztendlich eine Objekterkennungssoftware abgespielt. Diese Software gibt an einem, am Raspberry Pi angeschlossenen, Bildschirm ein Livebild der Kamera aus, umrandet erkannte Objekte und gibt zudem die Klassifizierung aus. Diese Dokumentation dient zur Veranschaulichung, Nachvollziehbarkeit und Reproduzierbarkeit des Projekts.

## 3 Benötigte Komponenten

||
|:--|
|Raspberry Pi 4|
|Stromversorgung für Raspberry Pi |
|Micro-SD-Karte (mind. 16 GB) |
|Raspberry Pi Kamera Modul (oder Webcam) |
|Maus und Tastatur |
|Bildschirm mit HDMI-Kabel (micro-HDMI zu HDMI)|
|Ggf. Raspberry Pi Gehäuse mit integrierter Kamera-Halterung|

## 4 Vorbereitung des Raspberry Pis

Um die Objekterkennung auf dem Raspberry Pi ausführen zu können, muss zunächst der Raspberry Pi vorbereitet werden. Es muss Raspberry OS auf der SD-Karte (Flashing) installiert werden und im Anschluss der Raspberry Pi konfiguriert werden.


Nach dem ersten Start des Raspberry Pis wird empfohlen dieses upzudaten. Hierfür ist folgender Befehl auszuführen:

    sudo apt-get update && sudo apt-get upgrade


Nun muss noch die Kameraschnittstelle aktiviert werden:

    sudo raspi-config

 
Im anschließend erscheinendem Menü zu „Interface-Options -> Camera“ navigieren um die Kameraschnittstelle zu aktivieren.

 
Wenn der Raspberry Pi aktualisiert ist, ist dieser bereit für die Implementierung des KI-Projekts.

## 5 Installation von TensorFlow 

### 5.1 Disclaimer 

TensorFlow ist eine Open Source-Plattform für maschinelles Lernen (Machine Learning). Es verfügt über ein umfassendes, flexibles System aus Tools, Bibliotheken und Community-Ressourcen, mit dem Forscher den Stand der Technik im maschinellen Lernen vorantreiben und Entwickler auf einfache Weise ML-basierte Anwendungen erstellen und bereitstellen können.

 
### 5.2 Python installieren

|Prüfen, ob Python bereits installiert ist.|
|:--|
|Mit dem Befehl `dpkg -l`  kann man sich alle installierten Packages des Raspberry Pis anzeigen lassen.  Sucht man ein bestimmtes Package empfiehlt sich der Befehl `apt-cache search (Package)`. In unserem Fall also `apt-cache search python` oder `apt-cache search python3`.|

 
Sollte Python noch nicht installiert sein, kann dies über den Befehl sudo apt install python3 idle3 getan werden.

 
### 5.3 TensorFlow Lite 2 installieren

#### 5.3.1 Voraussetzungen

TensorFlow benötigt einige Packages, die zuerst installiert werden müssen.

    sudo apt-get install -y libatlas-base-dev libhdf5-dev libc-ares-dev libeigen3-dev build-essential libsdl-ttf2.0-0 python-pygame festival python3-h5py


#### 5.3.2 Virtuelle Umgebung

Da es bei unterschiedlichen Abhängigkeiten zu den einzelnen Packages und Bibliotheken von Python zu Konflikten kommen kann, soll TensorFlow in einer virtuellen Umgebung ausgeführt werden, die diese Konflikte verhindern soll.

    pip3 install virtualenv Pillow numpy pygame


#### 5.3.3 rpi-vision installieren

Die Objekterkennung basiert auf einem Programm, welches von Leigh Johnson geschrieben wurde und das MobileNet V2 Modell zu Objekterkennung verwendet. Um diese nun zu installieren, müssen folgende Befehle in der Konsole eingegeben werden.

    cd ~
    git clone --depth 1 https://github.com/adafruit/rpi-vision.git
    cd rpi-vision
    python3 -m virtualenv -p $(which python3) .venv
    source .venv/bin/activate

 
#### 5.3.4 TensorFlow 2.x installieren

Durch den Befehl `python3 -m virtualenv -p $(which python3) .venv` wurde eine virtuelle Umgebung erstellt, die anschließend mit `source .venv/bin/activate` aktiviert wurde. Dass man sich nun in einer virtuellen Umgebung befindet, erkennt man an dem Hinweis `(.venv)` auf der linken Seite der Befehlszeile.

 
In dieser virtuellen Umgebung soll nun TensorFlow 2.3.1 installiert werden.

    wget https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/tensorflow-2.3.1-cp37-none-linux_armv7l_download.sh
    chmod a+x ./tensorflow-2.3.1-cp37-none-linux_armv7l_download.sh
    ./tensorflow-2.3.1-cp37-none-linux_armv7l_download.sh
    pip3 install --upgrade setuptools
    pip3 install tensorflow-*-linux_armv7l.whl
    pip3 install -e .

Nach erfolgreicher Installation muss das Raspberry Pi neugestartet werden.

    sudo reboot


### 5.4 Ausführen der Graphic Labeling Demo

Nun kann die Objekterkennung ausgeführt werden. Hierzu wird wieder zuerst eine virtuelle Umgebung aktiviert.

    cd rpi-vision && . .venv/bin/activate

Um das Programm mit einer einfachen vorgefertigten Objektbibliothek auszuführen, welches vorgehaltene Objekte auf dem Bildschirm ausgibt, wird folgender Befehl in der Konsole eingegeben

    python3 tests/pitft_labeled_output.py --tflite


### 5.5 Anpassen der Bildschirmausgabe

Beim ersten Starten der Objekterkennung wird die Bildschirmausgabe im Vollbildmodus ausgeführt. Dadurch ergibt sich das Problem, dass man das Programm nicht mehr durch den Befehl Strg + C beenden kann, da man keinen Zugriff mehr auf die Konsole hat. Um dieses Problem zu beheben, kann man im Programmcode die Größe des Ausgabefensters anpassen.

 
**Originalcode:**
```py
    # initialize  the display
    pygame.init() 
    screen =  pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    capture_manager  = None
```


**Angepasster Code:**
```py
    # initialize the display  
    pygame.init()  
    screen = pygame.display.set_mode((640,480)) # Breite 640 px, Höhe 480 px 
    capture_manager = None

```


## 6 Trainieren der KI

Zum Trainieren der KI wird "Teachable Machine" von Google verwendet.

> Teachable Machine ist ein webbasiertes Tool, mit dem sich Modelle für  maschinelles Lernen schnell und einfach erstellen lassen und das für  alle zugänglich ist.

Teachable Machine kann im Browser über den Link [https://teachablemachine.withgoogle.com/](https://teachablemachine.withgoogle.com/) aufgerufen werden. Nach Klick auf "Erste Schritte" zu einem Auswahlmenü, in dem man wählen kann, ob man eine bildbasierte, audiobasierte oder posenbasierte Erkennung trainieren möchte. Für unsere Zwecke nehmen wir die bildbasierte Erkennung über den Menüpunkt " Bildprojekt". Auf der nächsten Seite kann man die gewünschten Klassen definieren und entweder per Webcam oder Upload zur Verfügung stellen, auf deren Basis ein Modell trainiert werden soll.

![Screenshot 1](https://github.com/MKS84/Objekterkennung-TensorFlow-RaspberryPi/blob/main/Bilder/Bildschirmfoto%201.png)

Für ein besseres Ergebnis empfiehlt sich, Bilder vom zu erkennenden Objekt in verschiedenen Winkeln und unter verschiedenen Belichtungen der Software zur Verfügung zu stellen.

Mit Klick auf "Modell trainieren" beginnt die Software das Modell zu erstellen. Um das fertige Modell dann mit TensorFlow verwenden zu können, muss dieses unter "Modell exportieren" im Reiter "Tensorflow" als "Savedmodel" heruntergeladen werden.

![Screenshot 2](https://github.com/MKS84/Objekterkennung-TensorFlow-RaspberryPi/blob/main/Bilder/Bildschirmfoto%202.png)

## 7 Implementierung der Objekterkennung

Um das gespeicherte Archiv als zugrundeliegendes Modell der Objekterkennung in TensorFlow zu verwenden, müssen folgende Befehle in der Konsole eingegeben werden. Die heruntergeladene Datei "converted_savedmodel.zip" muss dabei im Verzeichnis "home/pi/" abliegen.

    $ cd rpi-vision 
    $ sudo bash 
    # source .venv/bin/activate 
    # python3 tests/pitft_teachablemachine.py ../converted_savedmodel.zip

![Screenshot 3](https://github.com/MKS84/Objekterkennung-TensorFlow-RaspberryPi/blob/main/Bilder/Braincraft.png)

## 8 Bewertung der KI

Die Trefferquote der Künstlichen Intelligenz hängt stark von den übergebenen Fotos ab. Die für das Trainieren der KI verwendeten Fotos sollten das zu erkennende Objekt in möglichst vielen Kamerawinkeln zeigen. Auch verschiedene Lichtverhältnisse und Bildhintergründe ermöglichen es die KI besser auf das Erkennen des gewünschten Objekts zu trainieren.

Im Rahmen dieses Projekts wurden mehrere Bildquellen für das Training der KI verwendet. Neben der Aufnahme von Fotos über die Webcam auf der Website von Teachable Machine wurden der KI mehrere Hundert Fotos von Objekten aus kostenlosen Fotokatalogen zur Verfügung gestellt. Es zeigte sich jedoch, dass selbst diese Anzahl wohl zu gering war.

Die besten Ergebnisse werden erreicht, wenn beim Testen der KI die gleichen Voraussetzungen wie beim Training der KI herrschen. So wurde die KI beispielsweise mithilfe von Teachable Machine darauf trainiert zu erkennen, ob eine Person vor der Kamera eine FFP2-Maske trägt oder nicht. Solange es sich um dieselbe Person und denselben Hintergrund wie aus den Trainingsdaten handelte, war die Trefferquote relativ hoch. Sobald jedoch eine andere Person ins Bild kam oder sich der Hintergrund veränderte, fiel die Trefferquote rapide ab.

Eine weitere Problematik, die sich hier ergibt, ist die Tatsache, dass man nicht genau sagen kann, wie die KI erkennt, ob die Person eine Maske trägt oder nicht. Für uns Menschen ist es relativ einfach zu sagen, wann welcher Fall gegeben ist, da es für uns einfach ist eine Maske zu erkennen. Die KI hingegen "kennt" keine Maske. Was sie vielleicht erkennt ist eine weiße Region in der Mitte des Bildes. Oder das Verhältnis der Helligkeit zwischen Bildmittelpunkt und Bildrand. Vielleicht "konzentriert" sich die KI gar nicht mal auf den Teil des Bildes, in dem die Maske des Trägers zu sehen ist, sondern auf einen völlig anderen Teil des eingelesenen Bildes. Das kann, wie in unserem Beispiel, dazu führen, dass selbst eine minimale Veränderung des Bildhintergrundes die Trefferquote verschlechtert.

Um die Trefferquote zu verbessern bräuchte man daher ein Vielzahl an unterschiedlichen Fotos, welche im besten Fall sogar so ausgewählt sind, dass das zu erkennende Merkmal besonders eindeutig präsent ist. Eine weitere Hilfe für die KI könnte es sein, den Bereich für die Erkennung einzugrenzen. In unserem Beispiel mit der Maske könnte man davon ausgehen, dass diese in den meisten Fällen in der Bildmitte zu finden ist, und so die Erkennung auf diesen Bereich fokusieren.

![Face Mask Detection](https://github.com/MKS84/Objekterkennung-TensorFlow-RaspberryPi/blob/main/Bilder/coronavirus-2019-ncov-covid-19-disease-prevention-concept-man-with-wearing-face-mask-without-mask-health-care-isolated-white-background-illustration_83111-597.jpg)

## 9 Literatur- und Abbildungsverzeichnis
 
### Websites

[https://www.tensorflow.org](https://www.tensorflow.org)

[https://learn.adafruit.com/running-tensorflow-lite-on-the-raspberry-pi-4/](https://learn.adafruit.com/running-tensorflow-lite-on-the-raspberry-pi-4/)

[https://learn.adafruit.com/teachable-machine-raspberry-pi-tensorflow-camera](https://learn.adafruit.com/teachable-machine-raspberry-pi-tensorflow-camera)

### Bilder

**Titelseite**

[https://images-na.ssl-images-amazon.com/images/I/71dCE5aNBCL._AC_SL1500_.jpg](https://images-na.ssl-images-amazon.com/images/I/71dCE5aNBCL._AC_SL1500_.jpg)

[https://1.bp.blogspot.com/-HKhrGghm3Z4/Xwd6oWNmCnI/AAAAAAAADRQ/Hff-ZgjSDvo7op7aUtdN--WSuMohSMn-gCLcBGAsYHQ/s1600/tensorflow2objectdetection.png](https://1.bp.blogspot.com/-HKhrGghm3Z4/Xwd6oWNmCnI/AAAAAAAADRQ/Hff-ZgjSDvo7op7aUtdN--WSuMohSMn-gCLcBGAsYHQ/s1600/tensorflow2objectdetection.png)

**Kapitel 9**

https://image.freepik.com/free-vector/coronavirus-2019-ncov-covid-19-disease-prevention-concept-man-with-wearing-face-mask-without-mask-health-care-isolated-white-background-illustration_83111-597.jpg 



