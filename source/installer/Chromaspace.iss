[Setup]
AppId={{A8CBEA7B-8D55-4A2E-8C23-5B8DCCF9D5A1}
AppName=Chromaspace
AppVersion=1.0.10 Beta
AppVerName=Chromaspace OFX v1.0.10 Beta
AppPublisher=Moaz ELgabry
AppPublisherURL=https://moazelgabry.com
AppSupportURL=https://github.com/MoazElgabry/ME_OFX/issues
AppUpdatesURL=https://github.com/MoazElgabry/ME_OFX
DefaultDirName={commoncf}\OFX\Plugins
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=Chromaspace_v1.0.10_Beta_Windows_Installer
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin

[Files]
Source: "..\bundle\Chromaspace.ofx.bundle\*"; DestDir: "{commoncf64}\OFX\Plugins\Chromaspace.ofx.bundle"; Flags: ignoreversion recursesubdirs createallsubdirs

[Code]
function ResolveRunning: Boolean;
var
  ResultCode: Integer;
begin
  Result := False;

  if Exec(
      ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      '-NoProfile -ExecutionPolicy Bypass -Command "if (Get-Process Resolve -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"',
      '',
      SW_HIDE,
      ewWaitUntilTerminated,
      ResultCode) then
  begin
    Result := (ResultCode = 1);
  end;
end;

function InitializeSetup(): Boolean;
var
  Clicked: Integer;
begin
  while ResolveRunning() do
  begin
    Clicked := SuppressibleMsgBox(
      'DaVinci Resolve is currently running.' + #13#10 +
      'Please close Resolve before installing Chromaspace.',
      mbError,
      MB_RETRYCANCEL,
      IDRETRY);

    if Clicked = IDCANCEL then
    begin
      Result := False;
      Exit;
    end;
  end;

  Result := True;
end;
