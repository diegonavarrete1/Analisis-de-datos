

-- 1
SELECT p.nombre, p.app, p.apm
FROM persona p
WHERE EXISTS (
    SELECT 1 FROM compra c WHERE c.idpersona = p.idpersona
);


SELECT p.nombre, p.app, p.apm
FROM persona p
WHERE p.nombre LIKE 'A%' 
AND p.anoNacimiento = 1988
AND EXISTS (
    SELECT 1 FROM compra c WHERE c.idpersona = p.idpersona
);

-- 8
SELECT c.numeroticket
FROM compra c
WHERE EXISTS (
    SELECT 1 
    FROM persona p 
    WHERE p.idpersona = c.idpersona 
    AND p.genero = 'M'
);
