import time
from jnius import autoclass
from android.runnable import run_on_ui_thread

# فایل اصلی تو را import می‌کنیم (بدون تغییر!)
from اخرین۷ import KhanjarSupremeV5

PythonActivity = autoclass('org.kivy.android.PythonActivity')
NotificationManager = autoclass('android.app.NotificationManager')
NotificationChannel = autoclass('android.app.NotificationChannel')
NotificationBuilder = autoclass('android.app.Notification$Builder')
Context = autoclass('android.content.Context')

class ScannerService:
    def __init__(self):
        self.running = False
        self.scanner = None
    
    def start(self, mActivity, argument):
        self.running = True
        self.scanner = KhanjarSupremeV5()
        
        # ایجاد کانال نوتیفیکیشن
        channel = NotificationChannel(
            'scanner_ch',
            'Crypto Scanner',
            NotificationManager.IMPORTANCE_HIGH
        )
        nm = mActivity.getSystemService(Context.NOTIFICATION_SERVICE)
        nm.createNotificationChannel(channel)
        
        # حلقه اصلی
        while self.running:
            try:
                # اجرای اسکنر اصلی (بدون تغییر!)
                self.scanner.run()
                
                # صبر 1 دقیقه
                time.sleep(60)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)
    
    def stop(self, mActivity):
        self.running = False

if __name__ == '__main__':
    from android import AndroidService
    service = AndroidService('Scanner Service', 'Running...')
    service.run()
