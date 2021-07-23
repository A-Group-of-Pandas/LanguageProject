import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.uix.image import Image, AsyncImage

class CanvasWidget(Widget):
    def __init__(self, **kwargs):
  
        super(CanvasWidget, self).__init__(**kwargs)
  
        # Arranging Canvas
        with self.canvas:
            Color(0.1, 1, 0.1, 0.9)  # set the color
            
            self.bind(pos = self.update_rect,
                  size = self.update_rect)
                  
            self.rect = Rectangle(pos = self.center,
                                  size =(self.width / 2.,
                                        self.height / 2.))
            
            self.bind(pos = self.update_rect,
                  size = self.update_rect)
  
    # update function which makes the canvas adjustable.
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

class window(App):
    def build(self):
        layout = GridLayout(cols=4, rows=4, size_hint_y=2)
        layout.add_widget(Label(text ="Hello World !", font_size ='20sp',
            color =[0.5, 0.1, 0.74, 1]))

        def search_images():
            pass
            
        image_button = Button(text="Search images")
        image_button.bind(on_press=search_images)

        layout.add_widget(image_button)
        
        layout.add_widget(CanvasWidget()) #
        
        return layout
    

# Here our class is initialized
# and its run() method is called. 
# This initializes and starts 
# our Kivy application.
if __name__=="__main__":
    window = ImageWindow()
    window().run() 


# Create a class of images so that we can show them. 
class ImageWindow(App):
    def build(self, path):
        img = Image(source=path)
        img_position = (250, 250)
        image_opacity = 0.5
        return img
    