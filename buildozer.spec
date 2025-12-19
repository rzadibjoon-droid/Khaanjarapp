[app]
title = Crypto Scanner
package.name = cryptoscanner
package.domain = org.khanjar

source.dir = .
source.include_exts = py

version = 1.0

requirements = python3,kivy==2.2.1,pyjnius,requests,numpy,android

orientation = portrait
fullscreen = 0

android.permissions = INTERNET,WAKE_LOCK,FOREGROUND_SERVICE,POST_NOTIFICATIONS
android.api = 31
android.minapi = 21
android.ndk = 25b
android.accept_sdk_license = True

services = Scanner:service.py:foreground

icon.filename = %(source.dir)s/icon.png
presplash.filename = %(source.dir)s/presplash.png

[buildozer]
log_level = 2
warn_on_root = 1
