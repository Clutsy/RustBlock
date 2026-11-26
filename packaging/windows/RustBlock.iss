; RustBlock Windows installer.
; Build with: iscc /DMyAppVersion=0.2.1 packaging\windows\RustBlock.iss

#define MyAppName "RustBlock"
#ifndef MyAppVersion
  #define MyAppVersion "0.2.1"
#endif
#define MyAppPublisher "Clutsy"
#define MyAppURL "https://github.com/Clutsy/RustBlock"
#define MyAppExeName "rustblock-gui.exe"

[Setup]
AppId={{7C4E1A52-3B7E-4D2A-9C1F-6B0A8F3D2E11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\RustBlock
DefaultGroupName=RustBlock
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=RustBlock-Setup-{#MyAppVersion}
SetupIconFile=rustblock.ico
UninstallDisplayIcon={app}\rustblock.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
LicenseFile=..\..\LICENSE
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "..\..\target\release\rustblock-gui.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\target\release\rustblock.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "rustblock.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\RustBlock"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\rustblock.ico"
Name: "{group}\Uninstall RustBlock"; Filename: "{uninstallexe}"
Name: "{autodesktop}\RustBlock"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\rustblock.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch RustBlock"; Flags: nowait postinstall skipifsilent
