'''Using ArcGIS implementation'''
def sinuosity(shape):
    channel = shape.length
    deltaX = shape.firstPoint.X-shape.lastPoint.X   #首端X坐标与末端X坐标差值
    deltaY = shape.firstPoint.Y-shape.lastPoint.Y   #首端Y坐标与末端Y坐标差值
    valley = math.sqrt(pow(deltaX, 2)+pow(deltaY, 2))
    return channel/valley
with arcpy.da.SearchCursor(fc, ['OID@', 'SHAPE@']) as cursor:
    for row in cursor:
        oid = row[0]
        shape = row[1]
        si = round(sinuosity(shape), 3)
        print(f'Stream ID {oid} has a sinuosity index of {si}')