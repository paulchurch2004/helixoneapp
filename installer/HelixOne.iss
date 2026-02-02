; HelixOne - Inno Setup Installer Script
; Professional Windows installer with auto-start, shortcuts, and clean uninstall

#define MyAppName "HelixOne"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "HelixOne Technologies"
#define MyAppURL "https://clever-conkies-89d13b.netlify.app/"
#define MyAppExeName "HelixOne.exe"
#define MyAppAssocName "HelixOne Trading Platform"
#define MyAppAssocExt ".hlx"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
; App identification
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Installation paths
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}

; Output settings
OutputDir=..\dist\installer
OutputBaseFilename=HelixOne_Setup_{#MyAppVersion}

; Compression (maximum)
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes

; Visual settings
SetupIconFile=..\assets\logo.ico
WizardStyle=modern
WizardImageFile=..\assets\wizard_image.bmp
WizardSmallImageFile=..\assets\wizard_small.bmp

; Privileges
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Uninstaller
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

; Misc
DisableProgramGroupPage=yes
DisableWelcomePage=no
DisableDirPage=no
AllowNoIcons=yes
ShowLanguageDialog=auto
ArchitecturesInstallIn64BitMode=x64compatible

; Version info embedded in Setup.exe
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Installer
VersionInfoCopyright=Copyright (C) 2025 {#MyAppPublisher}
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupicon"; Description: "Lancer HelixOne au demarrage de Windows"; GroupDescription: "Options supplementaires:"; Flags: unchecked

[Files]
; Main application files
Source: "..\dist\HelixOne\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Additional data files (if not included in PyInstaller bundle)
; Source: "..\data\formation_commerciale\*"; DestDir: "{app}\data\formation_commerciale"; Flags: ignoreversion recursesubdirs createallsubdirs
; Source: "..\ml_models\*"; DestDir: "{app}\ml_models"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "Lancer {#MyAppName}"
Name: "{autoprograms}\{#MyAppName}\Desinstaller {#MyAppName}"; Filename: "{uninstallexe}"

; Desktop (optional)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "Lancer {#MyAppName}"

; Quick Launch (optional)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Registry]
; Auto-start with Windows (if selected)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#MyAppName}"; ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; Tasks: startupicon

; App registration
Root: HKCU; Subkey: "Software\{#MyAppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\{#MyAppName}"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"; Flags: uninsdeletekey

[Run]
; Launch after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up user data on uninstall (optional - comment out to keep user data)
; Type: filesandordirs; Name: "{userappdata}\{#MyAppName}"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\cache"
Type: dirifempty; Name: "{app}"

[Code]
// Pascal Script for custom installation logic

var
  DownloadPage: TDownloadWizardPage;

// Check if app is running before install/uninstall
function IsAppRunning(): Boolean;
var
  ResultCode: Integer;
begin
  Result := False;
  if Exec('tasklist', '/FI "IMAGENAME eq {#MyAppExeName}" /NH', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    // Check if process is found
    Result := (ResultCode = 0);
  end;
end;

// Close running instance before install
procedure CloseRunningApp();
var
  ResultCode: Integer;
begin
  Exec('taskkill', '/F /IM {#MyAppExeName}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(1000); // Wait for process to terminate
end;

// Initialization
function InitializeSetup(): Boolean;
begin
  Result := True;

  // Check if app is running
  if IsAppRunning() then
  begin
    if MsgBox('{#MyAppName} est en cours d''execution. Voulez-vous le fermer pour continuer l''installation?',
              mbConfirmation, MB_YESNO) = IDYES then
    begin
      CloseRunningApp();
    end
    else
    begin
      Result := False;
    end;
  end;
end;

// Uninstall initialization
function InitializeUninstall(): Boolean;
begin
  Result := True;

  // Check if app is running
  if IsAppRunning() then
  begin
    if MsgBox('{#MyAppName} est en cours d''execution. Voulez-vous le fermer pour continuer la desinstallation?',
              mbConfirmation, MB_YESNO) = IDYES then
    begin
      CloseRunningApp();
    end
    else
    begin
      Result := False;
    end;
  end;
end;

// Custom wizard page for license (optional)
procedure InitializeWizard();
begin
  // Add custom branding or pages here if needed
end;

// After installation - register with Windows
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Post-installation tasks
    Log('Installation completed successfully');
  end;
end;
