echo 'const char *commit_date = "'`git show -s --format=%ci HEAD`'";'
echo 'const char *commit_hash = "'`git rev-parse HEAD`'";'
