import {Injectable} from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';
import {API_URL} from '../env';
import { EndSeason } from './endseason.model';

@Injectable()
export class EndSeasonApiService
{
    constructor(private http: HttpClient)
    {}

    getEndSeason()
    {
        return this.http
            .get<EndSeason[]>(`${API_URL}/seasonends`)
    }

}
