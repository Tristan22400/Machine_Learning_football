import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {API_URL} from '../env';
import { Calcul } from './calcul.model';

@Injectable()
export class DataApiService
{
    constructor(private http: HttpClient)
    {}

    getObjectives(team: string)
    {
        return this.http
            .get<any>(`${API_URL}/objectives/${team}`)
    }

    getCalculs(team: string)
    {
        return this.http
            .get<Calcul>(`${API_URL}/calculs/${team}`)
    }

}
