 echo "BUILD START"
 pip cache purge
 pip install --no-build-isolation -r requirements.txt
 python3.9 manage.py collectstatic --noinput --clear
 python3.9 manage.py migrate
 echo "BUILD END"
