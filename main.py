from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from android.permissions import request_permissions, Permission
from jnius import autoclass

PythonService = autoclass('org.kivy.android.PythonService')
Context = autoclass('android.content.Context')

class ScannerApp(App):
    def build(self):
        request_permissions([
            Permission.INTERNET,
            Permission.WAKE_LOCK,
            Permission.FOREGROUND_SERVICE,
            Permission.POST_NOTIFICATIONS
        ])
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        self.status_label = Label(
            text='📡 Scanner Ready',
            font_size='20sp',
            size_hint_y=0.3,
            color=(1, 1, 1, 1)
        )
        
        self.start_btn = Button(
            text='▶️ START',
            font_size='24sp',
            size_hint_y=0.35,
            background_color=(0.2, 0.8, 0.2, 1)
        )
        self.start_btn.bind(on_press=self.start_service)
        
        self.stop_btn = Button(
            text='⏹️ STOP',
            font_size='24sp',
            size_hint_y=0.35,
            background_color=(0.8, 0.2, 0.2, 1),
            disabled=True
        )
        self.stop_btn.bind(on_press=self.stop_service)
        
        layout.add_widget(self.status_label)
        layout.add_widget(self.start_btn)
        layout.add_widget(self.stop_btn)
        
        return layout
    
    def start_service(self, instance):
        self.status_label.text = '🟢 Running...'
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        PythonService.start(self.service, 'scanner service')
    
    def stop_service(self, instance):
        self.status_label.text = '🔴 Stopped'
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        PythonService.stop(self.service)

if __name__ == '__main__':
    ScannerApp().run()
