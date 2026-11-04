; Inno Setup 6 script: one Setup.exe from the PyInstaller folder (build.bat runs it with /DAppVersion=x.y.z).
; Installs for the current user (no administrator rights needed); the user can choose all users in the dialog.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "NSF Surface and Spectroscopy Analyzer"
#define AppShort "NSSA"
#define AppExe "NSSA.exe"

[Setup]
AppId={{26BAB918-ED73-4E48-989D-248C83221674}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Ed Nelson
DefaultDirName={autopf}\{#AppShort}
DefaultGroupName={#AppShort}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename=NSSA-Setup-{#AppVersion}
SetupIconFile=..\app_NSSA\assets\app_icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
Name: "associate"; Description: "Open .nid, .nhf and .gwy files with {#AppName}"; Flags: unchecked

[Files]
Source: "dist\NSSA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\colormaps"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; file types, only with the task 'associate' (removed again by the uninstaller)
Root: HKA; Subkey: "Software\Classes\NSSA.Image"; ValueType: string; ValueData: "AFM image (NSSA)"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\NSSA.Image\DefaultIcon"; ValueType: string; ValueData: "{app}\{#AppExe},0"; Tasks: associate
Root: HKA; Subkey: "Software\Classes\NSSA.Image\shell\open\command"; ValueType: string; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: associate
Root: HKA; Subkey: "Software\Classes\.nid\OpenWithProgids"; ValueType: string; ValueName: "NSSA.Image"; ValueData: ""; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\.nhf\OpenWithProgids"; ValueType: string; ValueName: "NSSA.Image"; ValueData: ""; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\.gwy\OpenWithProgids"; ValueType: string; ValueName: "NSSA.Image"; ValueData: ""; Flags: uninsdeletevalue; Tasks: associate

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent
