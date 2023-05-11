@echo off

git show -s --format=%%ci HEAD > tmpFile
set /p COMMIT_DATE= < tmpFile
del tmpFile

git rev-parse HEAD > tmpFile
set /p COMMIT_HASH= < tmpFile
del tmpFile

echo const char *commit_date = "%COMMIT_DATE%";
echo const char *commit_hash = "%COMMIT_HASH%";
