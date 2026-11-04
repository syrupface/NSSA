@echo off
rem Starts NSSA, the NSF Surface and Spectroscopy Analyzer. Optionally drop an image or spectroscopy file onto this file.
python "%~dp0app_NSSA\main.py" %*
